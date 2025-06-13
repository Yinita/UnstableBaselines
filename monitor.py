import ray, asyncio
from unstable.terminal_interface import TerminalInterface

ray.init(address="auto", namespace="unstable")  # connect to existing cluster (usually same node)

# Get handles to named actors
tracker = ray.get_actor("Tracker")
model_pool = ray.get_actor("ModelPool")
step_buffer = ray.get_actor("StepBuffer")
collector = ray.get_actor("Collector")

# Get VLLM actors from collector
# actors = ray.get(collector.get_actors.remote())

# Create and launch the terminal interface
# term = TerminalInterface.options(name="UI", num_cpus=0.5).remote(tracker, model_pool, actors=actors, step_buffer=step_buffer)
# term.run.remote()
print("init temrinal interface")
term = TerminalInterface(
    tracker=tracker,
    model_pool=model_pool,
    actors=[], #ray.get(collector.get_actors.remote()),
    step_buffer=step_buffer
)
asyncio.run(term.run())  # this will now print to your terminal!
