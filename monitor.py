import ray, asyncio
from unstable.terminal_interface import TerminalInterface

# connect to existing cluster (usually same node)
ray.init(address="auto", namespace="unstable")  

# Get handles to named actors
tracker = ray.get_actor("Tracker")
model_pool = ray.get_actor("ModelPool")
step_buffer = ray.get_actor("StepBuffer")
collector = ray.get_actor("Collector")

term = TerminalInterface(tracker=tracker, model_pool=model_pool, step_buffer=step_buffer)
asyncio.run(term.run())
