from typing import List, Tuple, Dict, Set
from collections import defaultdict, Counter
from tqdm import tqdm
import heapq


class GraphPatternMiner:
    def __init__(self, min_support: float = 0.01, window: int = 5,
                 max_pattern_length: int = 5, min_pattern_length: int = 2,
                 top_k: int = 1000) -> None:
        self.min_support = min_support
        self.window = window
        self.max_pattern_length = max_pattern_length
        self.min_pattern_length = min_pattern_length
        self.top_k = top_k
        self.patterns = []
        
        self.graph = defaultdict(lambda: defaultdict(int))
        self.item_counts = Counter()
        
    def mine_patterns(self, sequences: List[List[int]]) -> List[Tuple[Tuple[int, ...], int]]:
        total_sequences = len(sequences)
        min_support_count = max(1, int(self.min_support * total_sequences))
        
        print(f"\n🔍 Graph Pattern Mining")
        print(f"  Sequences: {total_sequences:,}")
        print(f"  Min support: {self.min_support} ({min_support_count} sequences)")
        print(f"  Window: {self.window}, Max length: {self.max_pattern_length}")
        
        # Step 1: Build co-occurrence graph
        self._build_graph(sequences, min_support_count)
        
        # Step 2: Extract patterns
        self._extract_patterns(min_support_count, total_sequences)
        
        # Step 3: Sort and return top-k
        self.patterns.sort(key=lambda x: x[1], reverse=True)
        self.patterns = self.patterns[:self.top_k]
        
        print(f"  ✓ Mined {len(self.patterns)} patterns\n")
        
        return self.patterns
    
    def _build_graph(self, sequences: List[List[int]], min_support_count: int) -> None:
        print(f"\n  Building co-occurrence graph...")
        
        # Count items and build edges
        for seq in tqdm(sequences, desc="  Processing sequences", 
                       ncols=100, unit="seq"):
            # Count unique items in sequence
            seen = set()
            for item in seq:
                if item > 0 and item not in seen:
                    self.item_counts[item] += 1
                    seen.add(item)
            
            # Build co-occurrence edges
            for i in range(len(seq)):
                if seq[i] == 0:
                    continue
                
                # Look ahead within window
                for j in range(i + 1, min(i + self.window, len(seq))):
                    if seq[j] == 0 or seq[i] == seq[j]:
                        continue
                    
                    # Add bidirectional edge
                    self.graph[seq[i]][seq[j]] += 1
                    self.graph[seq[j]][seq[i]] += 1
        
        # Filter by min support
        print(f"  🔧 Filtering graph by support...")
        filtered_graph = {}
        for src, neighbors in tqdm(self.graph.items(), desc="  Filtering nodes",
                                   ncols=100, unit="node"):
            filtered_neighbors = {
                tgt: cnt for tgt, cnt in neighbors.items()
                if cnt >= min_support_count
            }
            if filtered_neighbors:
                filtered_graph[src] = filtered_neighbors
        
        self.graph = filtered_graph
        
        num_nodes = len(self.graph)
        num_edges = sum(len(neighbors) for neighbors in self.graph.values())
        
        print(f"  ✓ Graph built: {num_nodes:,} nodes, {num_edges:,} edges")
    
    def _extract_patterns(self, min_support_count: int, total_sequences: int) -> None:
        print(f"\n  Extracting patterns...")
        self.patterns = []
        
        # Length 1: Single items
        if self.min_pattern_length <= 1:
            for item, count in tqdm(self.item_counts.items(), 
                                   desc="  Length-1 patterns",
                                   ncols=100, unit="pattern"):
                if count >= min_support_count:
                    self.patterns.append(((item,), count))
        
        # Length 2: Pairs from graph edges
        if self.min_pattern_length <= 2 and self.max_pattern_length >= 2:
            pair_patterns = []
            for src in tqdm(self.graph.keys(), desc="  Length-2 patterns",
                          ncols=100, unit="node"):
                for tgt, count in self.graph[src].items():
                    if src < tgt:  # Avoid duplicates
                        pair_patterns.append(((src, tgt), count))
            
            self.patterns.extend(pair_patterns)
        
        # Length 3+: Use priority queue for top-k
        if self.max_pattern_length >= 3:
            self._extract_longer_patterns(min_support_count)
        
        print(f"  ✓ Extracted {len(self.patterns)} patterns")
    
    def _extract_longer_patterns(self, min_support_count: int) -> None:
        print(f"  Mining longer patterns (length 3+)...")
        
        # Priority queue: (-support, pattern)
        pq = []
        visited = set()
        
        # Start from high-degree nodes
        node_degrees = [(len(neighbors), node) 
                       for node, neighbors in self.graph.items()]
        node_degrees.sort(reverse=True)
        
        # Limit exploration to top nodes
        top_nodes = [node for _, node in node_degrees[:min(200, len(node_degrees))]]
        
        for start_node in tqdm(top_nodes, desc="  Exploring patterns",
                              ncols=100, unit="node"):
            if start_node not in self.graph:
                continue
            
            # Try extending from this node
            for neighbor, edge_count in self.graph[start_node].items():
                if edge_count < min_support_count:
                    continue
                
                pattern = tuple(sorted([start_node, neighbor]))
                
                if len(pattern) < self.max_pattern_length:
                    self._extend_pattern_bfs(pattern, edge_count, 
                                            min_support_count, visited, pq)
        
        # Extract patterns from priority queue
        while pq and len(self.patterns) < self.top_k * 2:
            neg_support, pattern = heapq.heappop(pq)
            support = -neg_support
            
            if len(pattern) >= self.min_pattern_length:
                self.patterns.append((pattern, support))
    
    def _extend_pattern_bfs(self, current_pattern: Tuple[int, ...],
                           current_support: int, min_support_count: int,
                           visited: Set, pq: List) -> None:
        """Extend pattern using BFS"""
        if len(current_pattern) >= self.max_pattern_length:
            return
        
        pattern_key = tuple(sorted(current_pattern))
        if pattern_key in visited:
            return
        visited.add(pattern_key)
        
        # Find common neighbors
        common_neighbors = None
        for item in current_pattern:
            if item not in self.graph:
                return
            
            neighbors = set(self.graph[item].keys())
            if common_neighbors is None:
                common_neighbors = neighbors
            else:
                common_neighbors &= neighbors
        
        if not common_neighbors:
            return
        
        # Try extending with each common neighbor
        for new_item in common_neighbors:
            if new_item in current_pattern:
                continue
            
            # Estimate support (minimum edge count)
            min_edge_count = min(
                self.graph[item].get(new_item, 0)
                for item in current_pattern
            )
            
            if min_edge_count < min_support_count:
                continue
            
            extended_pattern = tuple(sorted(current_pattern + (new_item,)))
            extended_support = min(current_support, min_edge_count)
            
            # Add to priority queue (keep only top-k)
            if len(pq) < self.top_k * 3:
                heapq.heappush(pq, (-extended_support, extended_pattern))
            elif extended_support > -pq[0][0]:
                heapq.heapreplace(pq, (-extended_support, extended_pattern))
    
    def get_patterns(self) -> List[Tuple[Tuple[int, ...], int]]:
        """Get mined patterns"""
        return self.patterns
    
    def get_statistics(self) -> Dict:
        """Get mining statistics"""
        if not self.graph:
            return {}
        
        num_nodes = len(self.graph)
        num_edges = sum(len(neighbors) for neighbors in self.graph.values())
        avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
        
        degrees = [len(neighbors) for neighbors in self.graph.values()]
        max_degree = max(degrees) if degrees else 0
        
        # Pattern length distribution
        length_dist = Counter(len(p) for p, _ in self.patterns)
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'num_patterns': len(self.patterns),
            'pattern_length_distribution': dict(length_dist)
        }
