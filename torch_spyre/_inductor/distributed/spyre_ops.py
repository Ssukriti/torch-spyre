import torch


def spyre_all_reduce(x, *args, **kwargs):
    """
    Placeholder Spyre primitive.

    In real implementation this would emit
    a Spyre collective instruction.
    """

    print("Spyre lowering triggered for all_reduce")

    # This will be replaced by interface to Spyre comms all reduce
    # example: return comms.all_reduce(...)
    # For demo just return identity
    return x


## To DO: we should use async patterns instead
def spyre_all_reduce_async(x, *args, **kwargs):
    print("Spyre lowering triggered for all_reduce_async")
    # Return a fake async handle for demo
    return {"tensor": x}

def spyre_wait(handle):
    print("Spyre wait called for async handle")
    # If handle is an async handle dict, extract tensor
    if isinstance(handle, dict):
        return handle["tensor"]
    # Already a tensor, just pass through
    return handle