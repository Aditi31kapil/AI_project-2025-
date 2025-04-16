import streamlit as st
import numpy as np
import time
import heapq
from collections import deque
import pandas as pd
import os

# Set WebSocket timeout to prevent connection errors
os.environ['STREAMLIT_SERVER_WEBSOCKET_PING_TIMEOUT'] = '300'

# ----------------- Constants --------------------
CELL_TYPES = {
    0: ("white", "Free"),
    1: ("black", "Obstacle"),
    2: ("red", "Fire"),
    3: ("blue", "Hydrant"),
    4: ("green", "Finish"),
    5: ("orange", "Start")
}
GRID_SIZE = 10
MOVE_COST = 1
FILL_COST = 2
MAX_BUCKETS = 5
FIRE_BUCKETS = 2

# ----------------- Utility Functions --------------------
def load_grid(file):
    grid = []
    for line in file.readlines():
        grid.append(list(map(int, line.decode("utf-8").strip().split())))
    return np.array(grid)

def get_movement_arrows(path):
    """Tracks multiple movements through each cell with chronological order."""
    arrows_history = {}  # Maps cell position to list of arrows (newest first)
    
    for idx in range(1, len(path)):
        prev = path[idx - 1]
        curr = path[idx]
        
        # Determine direction of movement
        dx, dy = curr[0] - prev[0], curr[1] - prev[1]
        
        if dx == -1 and dy == 0:
            arrow = '↑'
        elif dx == 1 and dy == 0:
            arrow = '↓'
        elif dx == 0 and dy == -1:
            arrow = '←'
        elif dx == 0 and dy == 1:
            arrow = '→'
        else:
            arrow = ''
        
        # Add the arrow to the current cell's history (prepend to show newest first)
        if curr in arrows_history:
            arrows_history[curr].insert(0, arrow)
        else:
            arrows_history[curr] = [arrow]
            
    return arrows_history

def draw_grid(grid, path=[]):
    st.markdown("<h4>Environment</h4>", unsafe_allow_html=True)
    
    # Get arrow history for each cell
    arrows_history = get_movement_arrows(path)
    
    # Create a responsive grid with CSS
    grid_html = f'''
    <style>
    .grid-container {{
        display: grid;
        grid-template-columns: repeat({GRID_SIZE}, minmax(40px, 45px));
        gap: 1px;
        overflow-x: auto;
        padding-bottom: 10px;
    }}
    .grid-cell {{
        height: 45px;
        border: 1px solid #333;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        padding: 2px;
    }}
    </style>
    <div class="grid-container">
    '''
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            color, label = CELL_TYPES[grid[i][j]]
            cell_content = ""
            
            if (i, j) in arrows_history:
                arrow_stack = ""
                for idx, arrow in enumerate(arrows_history[(i, j)]):
                    font_size = 16 - (idx * 2)
                    if font_size < 8:
                        font_size = 8
                    
                    arrow_stack += f'<div style="line-height:0.9; font-size:{font_size}px;">{arrow}</div>'
                
                cell_content = arrow_stack
            
            grid_html += f'<div class="grid-cell" style="background-color:{color};">{cell_content}</div>'
    
    grid_html += '</div>'
    
    st.markdown(grid_html, unsafe_allow_html=True)

def neighbors(pos):
    x, y = pos
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            yield (nx, ny)

def find_positions(grid, value):
    return list(zip(*np.where(grid == value)))

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
def Euclidean_distance(pos1, pos2):
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
def heuristic(pos, goal):
    return manhattan_distance(pos, goal)  # or Euclidean_distance(pos, goal)
def heuristic_cost_estimate(start, goal):
    return manhattan_distance(start, goal)  # or Euclidean_distance(start, goal)

def animate_path(grid, path, delay=0.3):
    path_log = []
    # Update every N steps instead of every step to reduce WebSocket traffic
    steps_per_update = 3

    # Create a persistent placeholder for the grid
    grid_placeholder = st.empty()

    for i in range(1, len(path), steps_per_update):
        # Draw the grid with the current path segment
        with grid_placeholder:
            draw_grid(grid, path[:i+1])
        
        # Log steps for dataframe
        for j in range(max(1, i-steps_per_update), i+1):
            if j < len(path):
                path_log.append({"Step": j, "From": path[j-1], "To": path[j]})
        
        time.sleep(delay)
    
    # Final update with complete path
    with grid_placeholder:
        draw_grid(grid, path)
    
    return path_log

# ----------------- AI Algorithms --------------------
def bfs(grid, start, goal):
    queue = deque([(start, [start], 0)])  # (position, path, depth)
    visited = set([start])
    expanded = 0
    max_depth = 0

    while queue:
        current, path, depth = queue.popleft()
        expanded += 1
        max_depth = max(max_depth, depth)

        if current == goal:
            return path, expanded, max_depth

        for n in neighbors(current):
            if grid[n] != 1 and n not in visited:  # Not an obstacle
                visited.add(n)
                queue.append((n, path + [n], depth + 1))

    return [], expanded, max_depth

def dfs(grid, start, goal):
    stack = [(start, [start], 0)]  # (position, path, depth)
    visited = set()
    expanded = 0
    max_depth = 0

    while stack:
        current, path, depth = stack.pop()
        
        if current in visited:
            continue
            
        visited.add(current)
        expanded += 1
        max_depth = max(max_depth, depth)

        if current == goal:
            return path, expanded, max_depth

        for n in neighbors(current):
            if grid[n] != 1 and n not in visited:  # Not an obstacle
                stack.append((n, path + [n], depth + 1))

    return [], expanded, max_depth

def ucs(grid, start, goal):
    # (cost, position, path, depth)
    queue = [(0, start, [start], 0)]
    visited = set()
    expanded = 0
    max_depth = 0

    while queue:
        cost, current, path, depth = heapq.heappop(queue)
        
        if current in visited:
            continue
            
        visited.add(current)
        expanded += 1
        max_depth = max(max_depth, depth)

        if current == goal:
            return path, expanded, max_depth

        for n in neighbors(current):
            if grid[n] != 1 and n not in visited:  # Not an obstacle
                new_cost = cost + MOVE_COST
                heapq.heappush(queue, (new_cost, n, path + [n], depth + 1))

    return [], expanded, max_depth

def greedy(grid, start, goal):
    # (heuristic, position, path, depth)
    queue = [(manhattan_distance(start, goal), start, [start], 0)]
    visited = set()
    expanded = 0
    max_depth = 0

    while queue:
        _, current, path, depth = heapq.heappop(queue)
        
        if current in visited:
            continue
            
        visited.add(current)
        expanded += 1
        max_depth = max(max_depth, depth)

        if current == goal:
            return path, expanded, max_depth

        for n in neighbors(current):
            if grid[n] != 1 and n not in visited:  # Not an obstacle
                heapq.heappush(queue, (manhattan_distance(n, goal), n, path + [n], depth + 1))

    return [], expanded, max_depth

def a_star(grid, start, goal):
    # (f_score, g_score, position, path, depth)
    queue = [(manhattan_distance(start, goal), 0, start, [start], 0)]
    visited = set()
    expanded = 0
    max_depth = 0

    while queue:
        _, g_score, current, path, depth = heapq.heappop(queue)
        
        if current in visited:
            continue
            
        visited.add(current)
        expanded += 1
        max_depth = max(max_depth, depth)

        if current == goal:
            return path, expanded, max_depth

        for n in neighbors(current):
            if grid[n] != 1 and n not in visited:  # Not an obstacle
                new_g_score = g_score + MOVE_COST
                f_score = new_g_score + manhattan_distance(n, goal)
                heapq.heappush(queue, (f_score, new_g_score, n, path + [n], depth + 1))

    return [], expanded, max_depth
def best_first_search(grid, start, goal):
    """
    Best First Search Algorithm that uses a combination of Manhattan and Euclidean distances
    as a more sophisticated heuristic.
    """
    # Calculate initial heuristic (weighted combination of Manhattan and Euclidean)
    h_val = 0.7 * manhattan_distance(start, goal) + 0.3 * Euclidean_distance(start, goal)
    
    # (heuristic, position, path, depth)
    queue = [(h_val, start, [start], 0)]
    visited = set()
    expanded = 0
    max_depth = 0

    while queue:
        _, current, path, depth = heapq.heappop(queue)
        
        if current in visited:
            continue
            
        visited.add(current)
        expanded += 1
        max_depth = max(max_depth, depth)

        if current == goal:
            return path, expanded, max_depth

        for n in neighbors(current):
            if grid[n] != 1 and n not in visited:  # Not an obstacle
                # Combined heuristic for more nuanced evaluation
                h_val = 0.7 * manhattan_distance(n, goal) + 0.3 * Euclidean_distance(n, goal)
                heapq.heappush(queue, (h_val, n, path + [n], depth + 1))

    return [], expanded, max_depth
def dfs_id(grid, start, goal, max_depth=20):
    """
    Iterative Deepening DFS that visualizes exploration at each depth level.
    Returns (path, expanded, max_depth_reached)
    """
    total_expanded = 0
    viz_placeholder = st.empty()
    exploration_log = []
    
    # Create a progress bar for depth iterations
    progress_bar = st.progress(0)
    
    for depth_limit in range(max_depth):
        # Update progress bar
        progress_bar.progress((depth_limit + 1) / max_depth)
        st.write(f"Searching with depth limit: {depth_limit}")
        
        # Reset for this iteration
        visited = set([start])
        expanded = 0
        current_path = [start]
        
        # Store all nodes explored at this depth limit
        depth_exploration = {depth_limit: [start]}
        
        # Draw initial state
        with viz_placeholder:
            draw_exploration_state(grid, current_path, depth_exploration, depth_limit)
        
        # Perform depth-limited search
        result, expanded, path = depth_limited_search(
            grid, start, goal, depth_limit, visited, current_path,
            expanded, viz_placeholder, depth_exploration, depth_limit
        )
        
        total_expanded += expanded
        exploration_log.append({
            "depth_limit": depth_limit,
            "nodes_explored": expanded,
            "found_path": result is not None
        })
        
        # If we found a path to the goal
        if result:
            st.success(f"Found solution at depth limit {depth_limit}")
            return path, total_expanded, depth_limit
        
        # If no new nodes were explored, no need to go deeper
        if expanded == 0 and depth_limit > 0:
            st.warning("Search space exhausted - no deeper nodes to explore")
            break
    
    st.error(f"No path found within maximum depth {max_depth}")
    return [], total_expanded, max_depth

def depth_limited_search(grid, current, goal, depth_limit, visited, path_so_far,
                         expanded, viz_placeholder, exploration_history, current_depth_limit):
    """
    Recursive depth-limited search with visualization of search process.
    """
    # Count this node as expanded
    expanded += 1
    
    # Show current exploration state (every few steps to avoid WebSocket overload)
    if expanded % 5 == 0:  # Update visualization every 5 nodes
        with viz_placeholder:
            draw_exploration_state(grid, path_so_far, exploration_history, current_depth_limit)
            time.sleep(0.1)  # Brief pause to allow visualization
    
    # Check if we reached the goal
    if current == goal:
        # Show final state when goal is found
        with viz_placeholder:
            draw_exploration_state(grid, path_so_far, exploration_history, current_depth_limit, 
                                   highlight_path=path_so_far, found_goal=True)
        return path_so_far, expanded, path_so_far.copy()
    
    # If at depth limit, don't go deeper
    if depth_limit <= 0:
        return None, expanded, []
    
    # Track depth of nodes being explored
    current_depth = current_depth_limit - depth_limit
    if current_depth not in exploration_history:
        exploration_history[current_depth] = []
    exploration_history[current_depth].append(current)
    
    # Explore neighbors in order (for consistent visualization)
    for next_pos in sorted(neighbors(current)):
        if grid[next_pos] != 1 and next_pos not in visited:  # Not an obstacle and not visited
            visited.add(next_pos)
            path_so_far.append(next_pos)
            
            # Recursively search deeper
            result, new_expanded, found_path = depth_limited_search(
                grid, next_pos, goal, depth_limit - 1, visited, path_so_far,
                expanded, viz_placeholder, exploration_history, current_depth_limit
            )
            
            expanded = new_expanded  # Update expanded count
            
            if result:  # If we found a path
                return result, expanded, found_path
            
            # Backtrack - remove the last node as we're going back up
            path_so_far.pop()
    
    return None, expanded, []

def draw_exploration_state(grid, current_path, exploration_history, current_depth_limit, 
                          highlight_path=None, found_goal=False):
    """
    Visualize the current state of DFS-ID exploration.
    Shows:
    - Nodes explored at each depth (color-coded)
    - Current search path
    - Final path (if found)
    """
    # Generate a color map for different depths
    depth_colors = [
        "#ffcccc", "#ffb3b3", "#ff9999", "#ff8080", "#ff6666",
        "#ff4d4d", "#ff3333", "#ff1a1a", "#ff0000", "#e60000",
        "#cc0000", "#b30000", "#990000", "#800000", "#660000"
    ]
    
    # Create an HTML grid
    grid_html = """
    <style>
    .exploration-grid {
        display: grid;
        grid-template-columns: repeat(10, 40px);
        gap: 1px;
    }
    .cell {
        height: 40px;
        width: 40px;
        border: 1px solid #333;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        font-weight: bold;
    }
    .depth-label {
        position: absolute;
        top: 2px;
        right: 2px;
        font-size: 8px;
    }
    </style>
    <div class="exploration-grid">
    """
    
    # Current position being explored
    current_pos = current_path[-1] if current_path else None
    
    # Build the grid
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            pos = (i, j)
            cell_type = grid[i][j]
            color, label = CELL_TYPES[cell_type]
            cell_content = ""
            
            # Determine cell appearance based on exploration status
            if highlight_path and pos in highlight_path:
                # Final solution path
                color = "lime"
                path_idx = highlight_path.index(pos)
                cell_content = f"{path_idx}"
                
            elif pos == current_pos:
                # Current position being explored
                color = "purple"
                cell_content = "🔍"
                
            elif pos in current_path:
                # In current exploration path
                color = "orchid"
                path_idx = current_path.index(pos)
                cell_content = f"{path_idx}"
                
            else:
                # Check if this position was explored at any depth
                for depth, nodes in exploration_history.items():
                    if pos in nodes:
                        # Color based on exploration depth
                        depth_color_idx = min(depth, len(depth_colors)-1)
                        color = depth_colors[depth_color_idx]
                        cell_content = f'<span class="depth-label">d{depth}</span>'
                        break
            
            # Add the cell to the grid
            grid_html += f'<div class="cell" style="background-color:{color};">{cell_content}</div>'
    
    grid_html += "</div>"
    
    # Display exploration statistics
    stats_html = f"""
    <div style="margin-top:10px;">
        <h4>Exploration Stats - Current Depth Limit: {current_depth_limit}</h4>
        <ul>
    """
    
    total_explored = 0
    for depth, nodes in exploration_history.items():
        nodes_count = len(nodes)
        total_explored += nodes_count
        stats_html += f"<li>Depth {depth}: {nodes_count} nodes explored</li>"
    
    stats_html += f"""
        </ul>
        <p><b>Total nodes explored: {total_explored}</b></p>
        <p><b>Current path:</b> {" → ".join([str(p) for p in current_path])}</p>
    </div>
    """
    
    # Display the grid and stats
    st.markdown(grid_html + stats_html, unsafe_allow_html=True)

def beam_search(grid, start, goal, beam_width):
    """
    Beam Search Algorithm with configurable beam width.
    Only keeps the best 'beam_width' nodes at each depth level.
    """
    # Start with just the initial node
    beam = [(heuristic(start, goal), start, [start], 0)]
    visited = set([start])
    expanded = 0
    max_depth = 0
    current_depth = 0

    while beam:
        # Process all nodes at the current depth
        next_beam = []
        
        for _, current, path, depth in beam:
            expanded += 1
            max_depth = max(max_depth, depth)
            
            if current == goal:
                return path, expanded, max_depth
                
            # If we've moved to a new depth level
            if depth > current_depth:
                current_depth = depth
                
            # Generate all possible next nodes
            for n in neighbors(current):
                if grid[n] != 1 and n not in visited:  # Not an obstacle
                    h_val = heuristic(n, goal)
                    next_beam.append((h_val, n, path + [n], depth + 1))
                    visited.add(n)
        
        # If no nodes were generated, search failed
        if not next_beam:
            break
            
        # Keep only the best beam_width nodes
        next_beam.sort(key=lambda x: x[0])  # Sort by heuristic value
        beam = next_beam[:beam_width]  # Keep only the best beam_width nodes
    
    return [], expanded, max_depth

# --------------- Firefighter Simulation ---------------
def simulate_firefighter(grid, algorithm):
    start_time = time.time()
    
    # Find initial positions
    start_pos = find_positions(grid, 5)[0]
    hydrants = find_positions(grid, 3)
    fires = find_positions(grid, 2)
    finish = find_positions(grid, 4)[0]
    
    # Ensure start position is not a fire position
    if start_pos in fires:
        return [], 0, 0, 0, 0, "Error: Start position cannot be a fire position"
    
    # Choose the appropriate algorithm
    if algorithm == "BFS":
        search_fn = bfs
    elif algorithm == "DFS":
        search_fn = dfs
    elif algorithm == "UCS":
        search_fn = ucs
    elif algorithm == "Greedy":
        search_fn = greedy
    elif algorithm == "A*":
        search_fn = a_star
    elif algorithm == "Best First":
        search_fn = best_first_search
    elif algorithm == "Beam Search":
        # For Beam Search, we need to handle the beam width parameter
        def custom_beam_search(grid, start, goal):
            # Default beam width of 3, can be configurable via UI
            return beam_search(grid, start, goal, beam_width=3)
        search_fn = custom_beam_search
    elif algorithm == "DFS-ID":
        # Use a wrapper to adapt the interface
        def dfs_id_wrapper(grid, start, goal, max_depth=20):
            path, expanded, depth = dfs_id(grid, start, goal,max_depth=20)
            return path, expanded, depth
        search_fn = dfs_id_wrapper
    # Initialize firefighter state
    current_pos = start_pos
    water_buckets = 0
    remaining_fires = fires.copy()
    total_path = [current_pos]
    total_expanded = 0
    max_depth = 0
    total_cost = 0
    
    actions_log = []
    
    # Simulation loop
    while remaining_fires or current_pos != finish:
        # If we have insufficient water and fires remain, go to nearest hydrant
        if water_buckets < FIRE_BUCKETS and remaining_fires:
            # Use Manhattan distance to find closest hydrant
            closest_hydrant = min(hydrants, key=lambda h: manhattan_distance(current_pos, h))
            
            actions_log.append(f"Moving to nearest hydrant at {closest_hydrant} (Manhattan distance: {manhattan_distance(current_pos, closest_hydrant)})")
            
            # Find path to hydrant using the selected algorithm
            hydrant_path, expanded, depth = search_fn(grid, current_pos, closest_hydrant)
            if not hydrant_path:
                return [], 0, 0, 0, 0, "Failed to find path to hydrant"
            
            # Update state
            if len(hydrant_path) > 1:  # Only add if there's a path to follow
                total_path.extend(hydrant_path[1:])  # Skip first position to avoid duplication
                total_cost += len(hydrant_path) - 1  # Add cost of movement
            
            current_pos = closest_hydrant
            total_cost += FILL_COST  # Cost of filling water
            water_buckets = MAX_BUCKETS  # Fill maximum buckets
            total_expanded += expanded
            max_depth = max(max_depth, depth)
            
            actions_log.append(f"Filled {MAX_BUCKETS} buckets of water at hydrant (cost: {FILL_COST})")
            
        # If we have water and fires remain, go to nearest fire
        elif water_buckets >= FIRE_BUCKETS and remaining_fires:
            # Use Manhattan distance to find closest fire
            closest_fire = min(remaining_fires, key=lambda f: manhattan_distance(current_pos, f))
            
            actions_log.append(f"Moving to nearest fire at {closest_fire} (Manhattan distance: {manhattan_distance(current_pos, closest_fire)})")
            
            # Find path to fire using the selected algorithm
            fire_path, expanded, depth = search_fn(grid, current_pos, closest_fire)
            if not fire_path:
                return [], 0, 0, 0, 0, "Failed to find path to fire"
            
            # Update state
            if len(fire_path) > 1:  # Only add if there's a path to follow
                total_path.extend(fire_path[1:])  # Skip first position to avoid duplication
                total_cost += len(fire_path) - 1  # Add cost of movement
            
            current_pos = closest_fire
            water_buckets -= FIRE_BUCKETS  # Use buckets to extinguish fire
            remaining_fires.remove(closest_fire)
            total_expanded += expanded
            max_depth = max(max_depth, depth)
            
            actions_log.append(f"Extinguished fire using {FIRE_BUCKETS} buckets, {water_buckets} buckets remaining")
            
        # If no fires remain, go to finish
        elif not remaining_fires:
            actions_log.append(f"All fires extinguished. Moving to finish at {finish} (Manhattan distance: {manhattan_distance(current_pos, finish)})")
            
            # Find path to finish using the selected algorithm
            finish_path, expanded, depth = search_fn(grid, current_pos, finish)
            if not finish_path:
                return [], 0, 0, 0, 0, "Failed to find path to finish"
            
            # Update state
            if len(finish_path) > 1:  # Only add if there's a path to follow
                total_path.extend(finish_path[1:])  # Skip first position to avoid duplication
                total_cost += len(finish_path) - 1  # Add cost of movement
            
            current_pos = finish
            total_expanded += expanded
            max_depth = max(max_depth, depth)
            
            actions_log.append("Reached finish position")
            break
    
    execution_time = time.time() - start_time
    
    return total_path, total_expanded, max_depth, total_cost, execution_time, actions_log

# ----------------- Main Streamlit App --------------------
def main():
    st.set_page_config(page_title="Smart Fireman", layout="wide")
    st.title("🚒 Smart Firefighter Simulation")
    
    st.markdown("""
    This simulation models a firefighter who must:
    1. Fill water buckets from hydrants (max 5 buckets)
    2. Extinguish all fires (each fire needs 2 water buckets)
    3. Navigate to the finish position
    
    **Costs:**
    - Each move costs 1
    - Filling water costs 2
    
    **How it works:**
    - Uses Manhattan distance to find the nearest target
    - Uses the selected algorithm to plan the path to that target
    - This ensures sensible behavior while showing each algorithm's unique characteristics
    """)

    uploaded_file = st.file_uploader("Upload 10x10 Grid (.txt file)", type="txt")

    if uploaded_file:
        grid = load_grid(uploaded_file)
        
        # Ensure grid is valid
        if grid.shape != (GRID_SIZE, GRID_SIZE):
            st.error(f"Grid must be {GRID_SIZE}x{GRID_SIZE}")
        else:
            st.session_state.grid = grid
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                draw_grid(grid)
                st.write("Legend:")
                for value, (color, label) in CELL_TYPES.items():
                    st.markdown(f'<div style="display:flex;align-items:center;margin-bottom:5px;"><div style="background-color:{color};width:20px;height:20px;margin-right:10px;"></div> {value}: {label}</div>', unsafe_allow_html=True)
                
                algo = st.radio("Choose Algorithm", ("BFS", "DFS", "UCS", "Greedy", "A*","Best First", "Beam Search", "DFS-ID"))
                st.markdown("### Simulation Parameters")
                delay = st.slider("Animation Delay (seconds)", 0.0, 2.0, 0.2)
                # If Beam Search is selected, show the beam width slider
                if algo == "Beam Search":
                    beam_width = st.slider("Beam Width", 1, 10, 3, 
                                          help="Number of paths to explore at each level")
                    
                    # Update the Beam Search function to use the selected width
                    def custom_beam_search(grid, start, goal):
                        return beam_search(grid, start, goal, beam_width=beam_width)
                    
                    # Override the search function
                    if 'search_fn' in locals():
                        search_fn = custom_beam_search
                # Special options for DFS-ID
                if algo == "DFS-ID":
                    depth = st.slider("Maximum search depth", 1, 30, 15)
                    # Update the Beam Search function to use the selected width
                    def custom_depth_search(grid, start, goal):
                        return dfs_id(grid, start, goal, max_depth = depth)
                    
                    # Override the search function
                    if 'search_fn' in locals():
                        search_fn = custom_depth_search

                if st.button("Run Simulation"):
                    with st.spinner("Running simulation..."):
                        path, expanded, depth, cost, time_taken, actions = simulate_firefighter(grid, algo)
                        
                        if isinstance(actions, str):  # Error message
                            st.error(actions)
                        else:
                            with col2:
                                st.markdown("## 🔍 Simulation Log")
                                for i, action in enumerate(actions):
                                    st.write(f"{i+1}. {action}")
                                
                                st.markdown("### 🧾 Simulation Stats")
                                st.write(f"✅ **Total Path Cost:** {cost}")
                                st.write(f"📌 **Nodes Expanded:** {expanded}")
                                st.write(f"🌲 **Max Depth:** {depth}")
                                st.write(f"⏱️ **Time Taken:** {round(time_taken, 4)} seconds")
                            
                            # Animate the path
                            st.success("🚶 Starting Movement Simulation...")
                            path_log = animate_path(grid, path, delay=delay)
                            
                            # Display path as a dataframe
                            st.markdown("## 🔍 Movement Record")
                            df = pd.DataFrame(path_log)
                            st.dataframe(df)

if __name__ == "__main__":
    main()
