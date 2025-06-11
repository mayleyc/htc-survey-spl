import pandas as pd
import os

class TaxonomyParser():
    def __init__(self, file_path):
        self.file_path = file_path
        self.all_nodes = set()
        self.parent_nodes = set()
        self.child_to_parent = {}
        self.leaf_nodes = []

    def parse(self):
        # First pass: build parent-child relationships
        pass

    def _build_one_hot(self):
        # Final columns order: root first, then parents in order (excluding root), then leaves in order
        internal_parents = [p for p in self.parent_nodes if p != "root"]
        sorted_nodes = ["root"] + internal_parents + self.leaf_nodes
        node_to_index = {node: idx for idx, node in enumerate(sorted_nodes)}

        # Recursive ancestor getter
        def get_ancestors(node):
            ancestors = []
            while node in self.child_to_parent:
                node = self.child_to_parent[node]
                if node == "root":
                    break
                ancestors.append(node)
            return ancestors

        # Build one-hot for leaves only
        one_hot_dict = {}
        for leaf in self.leaf_nodes:
            vec = [0] * len(sorted_nodes)
            vec[node_to_index[leaf]] = 1
            for ancestor in get_ancestors(leaf):
                vec[node_to_index[ancestor]] = 1
            one_hot_dict[leaf] = vec

        return one_hot_dict
    
    def get_one_hot(self):
        return self._build_one_hot()

class AmazonTaxonomyParser(TaxonomyParser): # leaf-only lines allowed
    def parse(self):
        # First pass: build parent-child relationships
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue
                if len(tokens) > 1:
                    parent = tokens[0]
                    children = tokens[1:]

                    self.parent_nodes.add(parent)
                    self.all_nodes.add(parent)
                    self.all_nodes.update(children)

                    for child in children:
                        self.child_to_parent[child] = parent
                else:
                    leaf = tokens[0]
                    self.leaf_nodes.append(leaf)

class BGCParser(TaxonomyParser): # no leaf-only lines
    def parse(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue
                if len(tokens) > 1:
                    parent = tokens[0]
                    children = tokens[1:]

                    self.parent_nodes.add(parent)
                    self.all_nodes.add(parent)
                    self.all_nodes.update(children)

                    for child in children:
                        self.child_to_parent[child] = parent
                else:
                    continue

        self.leaf_nodes = [p for p in self.all_nodes if p not in self.parent_nodes]

'''def parse_taxonomy_amz(file_path):
    all_nodes = set()
    parent_nodes = set()
    child_to_parent = {}
    leaf_nodes = []

    # First pass: build parent-child relationships
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            if len(tokens) > 1:
                parent = tokens[0]
                children = tokens[1:]

                parent_nodes.add(parent)
                all_nodes.add(parent)
                all_nodes.update(children)

                for child in children:
                    child_to_parent[child] = parent
            else:
                leaf_nodes.append(tokens[0])


    # Final columns order: root first, then parents in order (excluding root), then leaves in order
    # Assuming root is first parent:
    internal_parents = [p for p in parent_nodes if p != "root"]
    sorted_nodes = ["root"] + internal_parents + leaf_nodes
    node_to_index = {node: idx for idx, node in enumerate(sorted_nodes)}

    # Recursive ancestor getter
    # Ancestor function but ignore root
    def get_ancestors(node):
        ancestors = []
        while node in child_to_parent:
            node = child_to_parent[node]
            if node == "root":
                break
            ancestors.append(node)
        return ancestors

    # Build one-hot for leaves only
    one_hot_dict = {}
    for leaf in leaf_nodes:
        vec = [0] * len(sorted_nodes)
        vec[node_to_index[leaf]] = 1
        for ancestor in get_ancestors(leaf):
            vec[node_to_index[ancestor]] = 1
        one_hot_dict[leaf] = vec

    return one_hot_dict, sorted_nodes

def parse_taxonomy_bgc(file_path):
    all_nodes = set()
    parent_nodes = set()
    child_to_parent = {}
    leaf_nodes = []

    # First pass: build parent-child relationships
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            if len(tokens) > 1:
                parent = tokens[0]
                children = tokens[1:]

                parent_nodes.add(parent)
                all_nodes.add(parent)
                all_nodes.update(children)

                for child in children:
                    child_to_parent[child] = parent
            else:
                continue


    # Final columns order: root first, then parents in order (excluding root), then leaves in order
    # Assuming root is first parent:
    leaf_nodes = [p for p in all_nodes if p not in parent_nodes]
    internal_parents = [p for p in parent_nodes if p != "root"]
    sorted_nodes = ["root"] + internal_parents + leaf_nodes
    node_to_index = {node: idx for idx, node in enumerate(sorted_nodes)}

    # Recursive ancestor getter
    # Ancestor function but ignore root
    def get_ancestors(node):
        ancestors = []
        while node in child_to_parent:
            node = child_to_parent[node]
            if node == "root":
                break
            ancestors.append(node)
        return ancestors

    # Build one-hot for leaves only
    one_hot_dict = {}
    for leaf in leaf_nodes:
        vec = [0] * len(sorted_nodes)
        vec[node_to_index[leaf]] = 1
        for ancestor in get_ancestors(leaf):
            vec[node_to_index[ancestor]] = 1
        one_hot_dict[leaf] = vec

    return one_hot_dict, sorted_nodes'''


def find_taxonomy_files(base_dir="dataset"):
    tax_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_tax.txt"):
                full_path = os.path.join(root, file)
                tax_files.append(full_path)
    return tax_files


# Example usage
if __name__ == "__main__":
    # Replace with taxonomy file path
    taxonomy_file = "data/Amazon/amazon_tax.txt"
    filename = taxonomy_file.split("/")[-1].split(".")[0]

    if "amazon" in filename.lower():
        parser = AmazonTaxonomyParser(taxonomy_file)
    elif "bgc" in filename.lower() or "wos" in filename.lower():
        parser = BGCParser(taxonomy_file)
    else:
        raise ValueError("Unsupported taxonomy file format. Use Amazon, BGC or WOS taxonomy files.")
    parser.parse()
    # Get one-hot encoding
    one_hot = parser.get_one_hot()

    # Save to CSV: only leaves as rows, all nodes as columns
    output_file = f"{filename}_one_hot.csv"
    pd.DataFrame.from_dict(one_hot, orient='index').to_csv(output_file)

    print(f"Leaf-only multi-hot encoding saved to {output_file}")
