
"""
Mazes a (registered as MazeA-v0) is adapted from:
	http://www.delorie.com/game-room/mazes/genmaze.cgi
with parameters N=4 and W=H=3.
Maze b, c are hand-picked.

All mazes has 'S' as a fixed start. Goals are randomly placed
at any empty cells ('S' or space).
"""

maze_layouts = {
	'a': """
+--+--+--+--+
|S |        |
|  |        |
+  +  +--+  +
|     |     |
|     |     |
+  +  +  +  +
|  |     |  |
|  |     |  |
+  +  +  +  +
|     |     |
|     |     |
+--+--+--+--+
""",
	'b': """
+--+--+--+--+
|          S|
|           |
+  +--+--+  +
|        |  |
|        |  |
+  +     +  +
|  |     |  |
|  |     |  |
+  +--+--+  +
|           |
|           |
+--+--+--+--+
""",
	'c': """
+--+--+--+--+
|           |
|           |
+  +--+--+  +
|  |     |  |
|  |     |  |
+  +     +  +
|  |     |  |
|  |     |  |
+  +--+--+  +
|           |
|S          |
+--+--+--+--+
""",

}
