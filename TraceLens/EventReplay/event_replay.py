###############################################################################
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Right now everything is very manually done, but maybe it can be improved
# checkout this as it might be useful: https://github.com/pytorch/pytorch/blob/main/torch/fx/operator_schemas.py

from pprint import pprint
from typing import Dict, Any, List, Optional, Tuple
import re
import copy
import ast

from .utils import (
    _get_torch_or_raise,
    TensorCfg,
    build_tensor,
    list_profile_tensor_types,
)


class EventReplayer:
    def __init__(
        self,
        event: Dict[str, Any],
        device: str = "cuda",
        lazy: bool = False,
        verbose: bool = False,
        *,
        input_semantics: str = "synthetic",
        input_values: Optional[Dict[str, Any]] = None,
        operator_schema: Optional[str] = None,
        max_storage_elements: int = 16_777_216,
    ):
        """
        Initialize the EventReplayer with the event data and device type.

        Args:
            event (Dict[str, Any]): From the pytorch profile json data['traceEvents']
            device (str): The device type ('cuda' or 'cpu').
            verbose (bool): Flag to enable verbose output.
        """
        self.event = copy.deepcopy(event)
        self.device = device
        self.lazy = lazy
        self.verbose = verbose
        from .safety import ReplayContractError, validate_event_metadata

        if input_semantics not in {"synthetic", "captured"}:
            raise ReplayContractError("input_semantics must be synthetic or captured")
        if input_semantics == "synthetic" and input_values is not None:
            raise ReplayContractError("Synthetic replay cannot accept captured values")
        if input_semantics == "captured" and input_values is None:
            raise ReplayContractError("Captured replay requires actual tensor values")
        validate_event_metadata(self.event, max_storage_elements)
        self.input_semantics = input_semantics
        self.input_values = input_values
        self.operator_schema = operator_schema
        self.max_storage_elements = max_storage_elements
        self._setup()

    def _setup(self):
        """
        Setup the event replayer by extracting relevant information from the event.
        """
        if self.verbose:
            print(f"Preparing {self.event['name']} event for replay")
        from .safety import operation_for_schema

        self.matched_schema = EventReplayer._search_schema(
            self.event, self.verbose, self.operator_schema
        )
        self._operation = operation_for_schema(self.matched_schema)
        self.event_replay_IR = EventReplayer._get_event_replay_IR(
            self.event, self.matched_schema, self.verbose
        )
        from .safety import validate_output_bound

        validate_output_bound(
            self._operation, self.event_replay_IR, self.max_storage_elements
        )
        if not self.lazy:
            if self.verbose:
                print("setting up args and kwargs")
            self.args, self.kwargs = self._materialize_inputs()

    def _materialize_inputs(self):
        if self.input_semantics == "synthetic":
            return self._get_args_kwargs(self.event_replay_IR, self.device)
        from .safety import validate_captured_inputs

        values = validate_captured_inputs(
            self.event_replay_IR, self.input_values, self.device
        )

        def value(entry):
            return (
                values[entry["arg_name"]]
                if isinstance(entry["value"], TensorCfg)
                else entry["value"]
            )

        return (
            [value(e) for e in self.event_replay_IR["list_pos_args"]],
            {e["arg_name"]: value(e) for e in self.event_replay_IR["list_kwargs"]},
        )

    def replay(self):
        """
        Replay the event using the matched schema and event replay IR.
        """
        torch = _get_torch_or_raise()
        # Call the function with the arguments
        if self.lazy:
            args, kwargs = self._materialize_inputs()
        else:
            args, kwargs = self.args, self.kwargs

        # Call the function with the arguments
        with torch.no_grad():
            return self._operation(*args, **kwargs)

    def verify(self, expected_output, *, rtol=1e-5, atol=1e-6):
        """Replay captured values and compare the isolated operator's output."""
        from .safety import ReplayContractError, compare_outputs

        if self.input_semantics != "captured":
            raise ReplayContractError(
                "Numerical verification against captured output requires captured inputs"
            )
        actual = self.replay()
        compare_outputs(actual, expected_output, rtol, atol)
        return actual

    @staticmethod
    def _search_schema(
        event: Dict[str, Any],
        verbose: bool = False,
        operator_schema: Optional[str] = None,
    ) -> Optional["torch._C.FunctionSchema"]:
        torch = _get_torch_or_raise()
        op_schemas = torch._C._jit_get_schemas_for_operator(event["name"])
        if operator_schema is not None:
            op_schemas = [s for s in op_schemas if str(s) == operator_schema]
        matches = []
        # print each schema in separate line
        if verbose:
            print(f"Found {len(op_schemas)} schemas for {event['name']}:")
            for schema in op_schemas:
                pprint(str(schema))
            print("-" * 80)

        for schema in op_schemas:
            if verbose:
                print(f"Checking schema:")
                pprint(str(schema))
            if EventReplayer._is_schema_match(event, schema, verbose):
                if verbose:
                    print(f"Schema matched successfully")
                    print("-" * 80)
                matches.append(schema)
            if verbose:
                print("-" * 80)

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            from .safety import ReplayContractError

            raise ReplayContractError(
                f"Ambiguous operator schema for {event['name']}; supply an exact operator_schema"
            )
        raise ValueError(
            f"Cannot find matching schema for {event['name']}. Please check the event data and schema."
        )

    @staticmethod
    def _is_schema_match(
        event: Dict[str, Any], schema: "torch._C.FunctionSchema", verbose: bool = False
    ) -> bool:
        """
        Check if the event matches the schema.

        Args:
            event (Dict[str, Any]): The event data.
            schema (torch._C.FunctionSchema): The schema to match against.

        Returns:
            bool: True if the event matches the schema, False otherwise.
        """
        op_name, pos_args_schema, kwargs_schema, return_type = (
            EventReplayer.parse_schema_string(schema)
        )
        full_args_schema = pos_args_schema + kwargs_schema
        # Check if the number of args in the event matches the schema
        if len(event["args"]["Input type"]) != len(full_args_schema):
            return False
        # Check if the types match
        for idx in range(len(event["args"]["Input type"])):
            profiled_type = event["args"]["Input type"][idx]
            schema_type = full_args_schema[idx]["arg_type"]
            # The real installed FunctionSchema exposes normalized types without
            # alias syntax. Retain support for callers passing schema strings.
            schema_type = re.sub(r"Tensor\([^)]*\)", "Tensor", schema_type)
            if verbose:
                print(f"Checking arg {idx}:")
                print(f"\tSchema type: {schema_type}")
                print(f"\tProfiled type: {profiled_type}")
            # Rules for matching types
            # 1. for tensor types, schema type should be 'Tensor' and profiled type can be any of the tensor types 'float', 'c10::Half', 'c10::BFloat16' ...
            # 2. for bool types, schema type should be 'bool' and profiled type is 'Scalar'. So we need to further check the concrete Inputs if it only contains 'true' or 'false'
            # 3. for int types, schema type should be 'int' or 'SymInt' and profiled type is 'Scalar'. So we need to further check the concrete Inputs if it is a digit
            # 4. for float types, schema type should be 'Scalar' and profiled type is 'Scalar'. So we need to further check the concrete Inputs if it is a float
            # 5. for int[] types, schema type should be 'int[]' or 'SymInt[]' and profiled type is 'ScalarList'. So we need to further check the concrete Inputs if it is a list of digits
            # 6. for bool[] types, schema type should be 'bool[]' and profiled type is 'ScalarList'. So we need to further check the concrete Inputs if it is a list of 'true' or 'false'
            # 7. for tensor[] types, we cannot replay the event as the tensor shapes are not provided in the event. So we need to skip this case. Maybe suggest PyTorch to add this in the future.
            is_match = True
            # if the schema type ends with '?' then the profiled type can be blank as well
            if schema_type.endswith("?"):
                schema_type = schema_type[:-1]
                if profiled_type == "":
                    continue
                elif (
                    profiled_type == "ScalarList"
                    and event["args"]["Concrete Inputs"][idx] == "[]"
                ):
                    continue
            if schema_type in ["Tensor", "Tensor?", "Tensor(a!)"]:
                if profiled_type not in list_profile_tensor_types:
                    is_match = False
            elif schema_type == "bool":
                profiled_value = event["args"]["Concrete Inputs"][idx]
                if profiled_value.lower() not in ["true", "false"]:
                    is_match = False
            elif schema_type == "int" or schema_type == "SymInt":
                if profiled_type != "Scalar":
                    is_match = False
                profiled_value = event["args"]["Concrete Inputs"][idx]
                if not profiled_value.lstrip("-").isdigit():
                    is_match = False
            elif schema_type in ["float", "Scalar"]:
                if profiled_type != "Scalar":
                    is_match = False
                profiled_value = event["args"]["Concrete Inputs"][idx]
                try:
                    float(profiled_value)
                except ValueError:
                    is_match = False
            elif schema_type == "str":
                if profiled_type not in {"Scalar", "str", "String"}:
                    is_match = False
            elif schema_type.startswith("int[") or schema_type.startswith("SymInt["):
                # custom dev debugging
                if profiled_type != "ScalarList":
                    is_match = False
                profiled_value = event["args"]["Concrete Inputs"][idx]
                if not (
                    profiled_value.strip().startswith("[")
                    and profiled_value.strip().endswith("]")
                ):
                    return False
                profiled_value_cleaned = (
                    [x.strip() for x in profiled_value.strip()[1:-1].split(",")]
                    if profiled_value.strip() != "[]"
                    else []
                )
                if not all(x.lstrip("-").isdigit() for x in profiled_value_cleaned):
                    is_match = False
            elif schema_type.startswith("bool["):
                if profiled_type != "ScalarList":
                    is_match = False
                profiled_value = event["args"]["Concrete Inputs"][idx]
                if not (
                    profiled_value.strip().startswith("[")
                    and profiled_value.strip().endswith("]")
                ):
                    return False
                profiled_value_cleaned = (
                    [x.strip() for x in profiled_value.strip()[1:-1].split(",")]
                    if profiled_value.strip() != "[]"
                    else []
                )
                if not all(
                    x.lower() in ["true", "false"] for x in profiled_value_cleaned
                ):
                    is_match = False
            elif schema_type.startswith("Tensor["):
                is_match = False
            else:
                # raise ValueError(f"Unknown schema type: {schema_type}")
                # warning: if the schema type is not in the list, we will skip this case
                is_match = False
            if not is_match:
                if verbose:
                    print(
                        f"Schema type {schema_type} does not match profiled type {profiled_type}"
                    )
                return False
        return True

    @staticmethod
    def _get_event_replay_IR(
        event: Dict[str, Any], schema: "torch._C.FunctionSchema", verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Get the event replay IR from the event and schema.

        Args:
            event (Dict[str, Any]): The event data.
            schema (torch._C.FunctionSchema): The schema to match against.

        Returns:
            {
                'pos_args': [

                    dummy_tensor0,
                    dummy_tensor1,
                    value0,
                    value1,
                    ...
                ],
                'kwargs': {
                    'arg0': value0,
                    'arg1': dummy_tensor0,
                    'arg2': dummy_tensor1,
                    'arg3': value1,
                    ...
                }
            }
        """
        evt_name = event["name"]
        op_name, pos_args_schema, kwargs_schema, return_type = (
            EventReplayer.parse_schema_string(schema)
        )
        full_args_schema = pos_args_schema + kwargs_schema
        list_pos_args = []
        list_kwargs = []
        for idx in range(len(event["args"]["Input type"])):
            arg_name = full_args_schema[idx]["arg_name"]
            arg_type = full_args_schema[idx]["arg_type"]
            normalized_type = re.sub(r"Tensor\([^)]*\)", "Tensor", arg_type)
            base_type = normalized_type.rstrip("?")

            if verbose:
                print(f"Processing arg {idx}: {arg_name} ({arg_type})")
                print(f"Profiled args type: {event['args']['Input type'][idx]}")
                print(f"Profiled args dims: {event['args']['Input Dims'][idx]}")
                print(f"Profiled args strides: {event['args']['Input Strides'][idx]}")
                print(f"Concrete Inputs: {event['args']['Concrete Inputs'][idx]}")

            if arg_type.endswith("?") and event["args"]["Input type"][idx] == "":
                value = None
            elif (
                normalized_type.endswith("?")
                and event["args"]["Concrete Inputs"][idx] == "[]"
                and not base_type.endswith("]")
            ):
                value = None
            else:
                if base_type == "Tensor":
                    init = "normal"
                    if event["args"]["Input type"][idx] in {"bool", "int", "long"}:
                        init = "zeros"
                    elif evt_name == "aten::fill_":
                        # special case for fill_ where we don't need to initialize the tensor
                        # as it will be filled with a value later
                        init = None
                    elif evt_name == "aten::copy_" and arg_name != "src":
                        # special case for copy_ where we don't need to initialize the tensor
                        # as it will be copied from another tensor
                        init = None
                    value = TensorCfg(
                        shape=event["args"]["Input Dims"][idx],
                        dtype=event["args"]["Input type"][idx],
                        strides=event["args"]["Input Strides"][idx],
                        init=init,
                    )
                else:
                    arg_str = event["args"]["Concrete Inputs"][idx]
                    if base_type == "bool":
                        value = arg_str.lower() == "true"
                    elif base_type in ["int", "SymInt"]:
                        value = int(arg_str)
                    elif base_type in ["float", "Scalar"]:
                        value = (
                            int(arg_str)
                            if base_type == "Scalar" and arg_str.lstrip("-").isdigit()
                            else float(arg_str)
                        )
                    elif base_type == "str":
                        value = (
                            ast.literal_eval(arg_str)
                            if arg_str.startswith(("'", '"'))
                            else arg_str
                        )
                    elif base_type.startswith("int[") or base_type.startswith(
                        "SymInt["
                    ):
                        value = (
                            [int(x.strip()) for x in arg_str.strip()[1:-1].split(",")]
                            if arg_str.strip() != "[]"
                            else []
                        )
                    elif base_type.startswith("bool["):
                        value = (
                            [
                                x.strip().lower() == "true"
                                for x in arg_str.strip()[1:-1].split(",")
                            ]
                            if arg_str.strip() != "[]"
                            else []
                        )
                    else:
                        raise ValueError(f"Unsupported arg type: {arg_type}")
            if verbose:
                print(f"Parsed value: {value}")
                print(
                    f"Positional/Keyword: {'Positional' if idx < len(pos_args_schema) else 'Keyword'}"
                )
                print("-" * 80)
            if idx < len(pos_args_schema):
                list_pos_args.append(
                    {"arg_name": arg_name, "arg_type": arg_type, "value": value}
                )
            else:
                list_kwargs.append(
                    {"arg_name": arg_name, "arg_type": arg_type, "value": value}
                )
        return {"list_pos_args": list_pos_args, "list_kwargs": list_kwargs}

    @staticmethod
    def _get_args_kwargs(
        event_replay_IR: Dict[str, Any], device: str = "cuda"
    ) -> tuple[List["torch.Tensor"], Dict[str, Any]]:
        """
        Get the arguments and keyword arguments from the event replay IR.

        Args:
            event_replay_IR (Dict[str, Any]): The event replay IR.

        Returns:
            (List[torch.Tensor], Dict[str, Any]): The positional arguments and keyword arguments.
        """
        pos_args = []
        for arg in event_replay_IR["list_pos_args"]:
            value = arg["value"]
            if isinstance(value, TensorCfg):
                pos_args.append(build_tensor(value, device=device))
            else:
                pos_args.append(value)
        kwargs = {}
        for arg in event_replay_IR["list_kwargs"]:
            value = arg["value"]
            if isinstance(value, TensorCfg):
                kwargs[arg["arg_name"]] = build_tensor(value, device=device)
            else:
                kwargs[arg["arg_name"]] = value
        return pos_args, kwargs

    @staticmethod
    def parse_schema_string(
        schema,
    ) -> Tuple[str, List[Tuple[str, str, Optional[str], bool]], str]:
        if hasattr(schema, "arguments"):
            args, kwargs = [], []
            for argument in schema.arguments:
                entry = {
                    "arg_type": argument.type.str(),
                    "arg_name": argument.name,
                    "default": (
                        argument.default_value if argument.has_default_value() else None
                    ),
                }
                (kwargs if argument.kwarg_only else args).append(entry)
            return schema.name, args, kwargs, str(schema.returns)
        schema_str = str(schema)
        match = re.match(r"^([^\(]+)\((.*)\)\s*->\s*(.*)$", schema_str.strip())
        if not match:
            raise ValueError(f"Cannot parse schema string: {schema_str}")
        op_name, args_str, return_type = match.groups()
        parts = args_str.split("*")
        pos_part = parts[0].rstrip(",").strip()
        kwarg_part = parts[1].lstrip(",").strip() if len(parts) > 1 else ""

        def _parse_arg(raw_arg: str) -> Tuple[str, str, Optional[str], bool]:
            m = re.match(r"^(\S+)\s+(.*)$", raw_arg)
            if not m:
                raise ValueError(f"Invalid arg: {raw_arg}")
            arg_type, rest = m.groups()
            m2 = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(?:=(.*))?$", rest)
            if not m2:
                raise ValueError(f"Invalid arg name/default: {rest}")
            arg_name, default = m2.group(1), (
                m2.group(2).strip() if m2.group(2) else None
            )
            return arg_type.strip(), arg_name.strip(), default

        args = []
        for item in [x.strip() for x in pos_part.split(",") if x.strip()]:
            arg_type, arg_name, default = _parse_arg(item)
            args.append(
                {"arg_type": arg_type, "arg_name": arg_name, "default": default}
            )
        kwargs = []
        for item in [x.strip() for x in kwarg_part.split(",") if x.strip()]:
            arg_type, arg_name, default = _parse_arg(item)
            kwargs.append(
                {"arg_type": arg_type, "arg_name": arg_name, "default": default}
            )

        return op_name.strip(), args, kwargs, return_type.strip()

    def get_repro_info(self) -> Dict[str, Any]:
        """
        Extracts the minimal, serializable information needed to reproduce the event call.

        Returns:
            Dict[str, Any]: A dictionary containing the operator name and the replay IR.
                            Suitable for JSON serialization using the custom encoder.
        """
        # return {
        #     'op_name': self.event['name'],
        #     'replay_ir': self.event_replay_IR
        #     # No device info here - device is decided by the runner
        # }
        dict_repro_info = {}
        dict_repro_info["op_name"] = self.event["name"]
        list_pos_args, list_kwargs = (
            self.event_replay_IR["list_pos_args"],
            self.event_replay_IR["list_kwargs"],
        )
        # Convert TensorCfg to dict for JSON serialization
        list_pos_args_copy, list_kwargs_copy = copy.deepcopy(
            list_pos_args
        ), copy.deepcopy(list_kwargs)
        for idx, val in enumerate(list_pos_args_copy):
            if isinstance(val["value"], TensorCfg):
                list_pos_args_copy[idx]["value"] = val["value"].__dict__
        for idx, val in enumerate(list_kwargs_copy):
            if isinstance(val["value"], TensorCfg):
                list_kwargs_copy[idx]["value"] = val["value"].__dict__
        dict_repro_info["replay_ir"] = {
            "list_pos_args": list_pos_args_copy,
            "list_kwargs": list_kwargs_copy,
        }
        dict_repro_info["operator_schema"] = str(self.matched_schema)
        dict_repro_info["input_semantics"] = self.input_semantics
        dict_repro_info["event"] = {
            "cat": "cpu_op",
            "name": self.event["name"],
            "args": {
                key: copy.deepcopy(self.event["args"][key])
                for key in (
                    "Input Dims",
                    "Input Strides",
                    "Input type",
                    "Concrete Inputs",
                )
            },
        }
        if self.input_semantics == "captured":
            # IR describes layouts; it never serializes the caller's tensor values.
            dict_repro_info["requires_captured_values"] = True
        return dict_repro_info
