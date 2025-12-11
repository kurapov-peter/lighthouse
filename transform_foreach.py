#!/usr/bin/env python3
from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import interpreter, structured

import argparse

CONTRACT_MATCHER = "__match_contract"
CONTRACT_ACTION = "__action_contract"


def build_schedule() -> ir.Module:
    anytype = transform.AnyOpType.get()
    schedule = ir.Module.create()
    schedule.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()

    with ir.InsertionPoint(schedule.body):
        contract_matcher = transform.named_sequence(
            CONTRACT_MATCHER,
            [anytype],
            [anytype],
            arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
        )

    with ir.InsertionPoint(contract_matcher.body):
        transform.match_operation_name(contract_matcher.bodyTarget, {"linalg.contract"})
        transform.yield_([contract_matcher.bodyTarget])

    with ir.InsertionPoint(schedule.body):
        transform_action = transform.named_sequence(
            CONTRACT_ACTION,
            [transform.OperationType.get("linalg.contract")],
            [],
            arg_attrs=[{"transform.consumed": ir.UnitAttr.get()}],
        )

    with ir.InsertionPoint(transform_action.body):
        transform.PrintOp(
            target=transform_action.bodyTarget, name="action:transform_action"
        )
        transform.yield_()

    with ir.InsertionPoint(schedule.body):
        named_seq = transform.named_sequence(
            "__transform_entry_point",
            [transform.AnyOpType.get()],
            [],
            arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
        )

    with ir.InsertionPoint(named_seq.body):
        anytype = transform.AnyOpType.get()
        root = named_seq.bodyTarget

        func = structured.MatchOp.match_op_names(root, ["func.func"])
        transform.PrintOp(target=func, name="before transformations")

        matcher_refs = ir.ArrayAttr.get([ir.FlatSymbolRefAttr.get(CONTRACT_MATCHER)])
        action_refs = ir.ArrayAttr.get([ir.FlatSymbolRefAttr.get(CONTRACT_ACTION)])

        transform.foreach_match(
            updated=anytype,
            forwarded_outputs=[],
            root=func,
            forwarded_inputs=[],
            matchers=matcher_refs,
            actions=action_refs,
        )

        transform.yield_()

    return schedule


def apply_schedule() -> None:
    parser = argparse.ArgumentParser(
        description="Apply transform schedule to an input MLIR file."
    )
    parser.add_argument("input", type=str, help="Input MLIR file")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        input_module = ir.Module.parse(f.read())

    schedule = build_schedule()
    print("// ---- Schedule IR ----")
    print(schedule)
    print("\n\n")

    interpreter.apply_named_sequence(
        payload_root=input_module,
        transform_root=schedule.body.operations[2],
        transform_module=schedule,
    )

    print("// ---- Transformed Module ----")
    print(input_module)


if __name__ == "__main__":
    with ir.Context() as ctx, ir.Location.unknown():
        apply_schedule()
