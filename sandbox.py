from nvitop.tui.library import BufferedHistoryGraph
import numpy as np
import time 

graph=BufferedHistoryGraph(100.0, width=6, height=3, format=lambda v:f"{v:.0f}%") #,dynamic_bound=False)
    
for _ in range(100):
    time.sleep(1)
    print("\n\n\n")
    graph.add(value=100*np.random.uniform())
    print('\n'.join(graph.graph))
# graph.add(100*np.random.uniform())
# graph.add(100*np.random.uniform())
# graph.add(100*np.random.uniform())
graph.remake_graph()
print(graph.last_graph)
print(graph.graph)
# print(graph.last_value)
# print(graph.graph)
body = '\n'.join(graph.last_graph)
print(body)