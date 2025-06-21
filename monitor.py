import ray, asyncio
from unstable.terminal_interface import TerminalInterface

# connect to existing cluster (usually same node)
ray.init(address="auto", namespace="unstable")  

# Get handles to named actors
tracker = ray.get_actor("Tracker")
step_buffer = ray.get_actor("StepBuffer")

term = TerminalInterface(tracker=tracker, step_buffer=step_buffer)
asyncio.run(term.run())
