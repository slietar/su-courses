# Testing

if 0:
  graph = Graph.random(15, 0.3)
  # pickle.dump(graph, open("graph.pickle", "wb"))
  # graph: Graph = pickle.load(open("graph.pickle", "rb"))

  # print(cover_from_coupling(graph))
  # print(cover_greedy(graph))
  print(cover_optimal2(graph))
  print(cover_optimal3(graph))
  print(cover_optimal4(graph))

  with (Path(__file__).parent / "out.svg").open("wt") as file:
    file.write(graph.draw())


if 0:
  sample_count = 1000

  total_explored_node_count1 = 0
  total_explored_node_count2 = 0
  total_explored_node_count3 = 0

  for _ in range(sample_count):
    graph = Graph.random(12, 0.3)

    solutions = cover_optimal1(graph)
    cover1 = cover_optimal2(graph)
    cover2 = cover_optimal3(graph)
    cover3 = cover_optimal4(graph)

    total_explored_node_count1 += cover1[0]
    total_explored_node_count2 += cover2[0]
    total_explored_node_count3 += cover3[0]

    try:
      assert cover1[1] in solutions
      assert cover2[1] in solutions
      assert cover3[1] in solutions
    except:
      print(solutions)

      with (Path(__file__).parent / "out.svg").open("wt") as file:
        file.write(graph.draw())

      raise

  print(total_explored_node_count1 / sample_count)
  print(total_explored_node_count2 / sample_count)
  print(total_explored_node_count3 / sample_count)


if 0:
  sample_count = 100
  total_explored_node_count = 0

  for i in range(sample_count):
    print(f"> {i}")
    graph = Graph.random(20, 0.3)
    explored_node_count, _ = cover_optimal2(graph)
    total_explored_node_count += explored_node_count

  print(total_explored_node_count / sample_count)
