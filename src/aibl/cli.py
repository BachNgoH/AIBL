"""Public AIBL commands."""

def train():
    from aibl.train import main
    return main()

def infer():
    from aibl.inference import main
    return main()

def evaluate():
    from aibl.evaluate import main
    return main()
