//! Tree data structure — Rust-accelerated for common operations.

use std::fmt;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// Tree

#[pyclass(name = "Tree", module = "fastnltk._rust")]
#[derive(Clone)]
pub struct Tree {
    label: String,
    children: Vec<TreeNode>,
}

#[derive(Clone)]
enum TreeNode {
    Leaf(String),
    Subtree(Tree),
}

#[pymethods]
impl Tree {
    #[new]
    #[pyo3(signature = (label, children=None))]
    fn new(label: &str, children: Option<Vec<String>>) -> Self {
        let tree_children = children.unwrap_or_default().into_iter().map(TreeNode::Leaf).collect();
        Self { label: label.to_string(), children: tree_children }
    }

    /// Create from NLTK bracket string: "(S (NP The) (VP runs))"
    #[staticmethod]
    fn from_string(string: &str) -> PyResult<Self> {
        let s = string.trim();
        if !s.starts_with('(') {
            return Err(PyValueError::new_err("Tree string must start with '('"));
        }
        Self::parse_brackets(s, 0).map(|(tree, _)| tree).map_err(PyValueError::new_err)
    }

    fn __str__(&self) -> String {
        format!("{self}")
    }

    fn __repr__(&self) -> String {
        format!("Tree(\"{}\", {})", self.label, self.children_repr())
    }

    fn label(&self) -> String {
        self.label.clone()
    }

    #[must_use]
    fn leaves(&self) -> Vec<String> {
        let mut result = Vec::new();
        self.collect_leaves(&mut result);
        result
    }

    fn leaf_treepositions(&self) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        self.collect_treepositions(&mut vec![], &mut result);
        result
    }

    fn height(&self) -> usize {
        let mut max_child = 0;
        for child in &self.children {
            match child {
                TreeNode::Leaf(_) => max_child = max_child.max(1),
                TreeNode::Subtree(t) => max_child = max_child.max(t.height()),
            }
        }
        1 + max_child
    }

    /// Number of leaf tokens (matches NLTK's `len(tree)`).
    fn __len__(&self) -> usize {
        self.leaves().len()
    }

    /// Get child by index. Returns a `Tree` for non-leaf children, `str` for leaves.
    #[allow(deprecated)]
    fn __getitem__(&self, py: Python<'_>, index: isize) -> Py<PyAny> {
        let idx = if index < 0 {
            (self.children.len() as isize + index) as usize
        } else {
            index as usize
        };
        if idx >= self.children.len() {
            return py.None();
        }
        match &self.children[idx] {
            TreeNode::Leaf(s) => s
                .clone()
                .into_pyobject(py)
                .map_or_else(|_| py.None(), |obj| obj.into_any().unbind()),
            TreeNode::Subtree(t) => {
                Py::new(py, t.clone()).map_or_else(|_| py.None(), pyo3::Py::into_any)
            }
        }
    }

    /// Iterate over children (returns leaf strings / subtree bracket strings).
    fn __iter__(&self) -> Vec<String> {
        self.children
            .iter()
            .map(|c| match c {
                TreeNode::Leaf(s) => s.clone(),
                TreeNode::Subtree(t) => format!("{t}"),
            })
            .collect()
    }

    #[must_use]
    fn productions(&self) -> Vec<String> {
        let mut prods = Vec::new();
        self.collect_productions(&mut prods);
        prods
    }

    /// Return all subtrees as Rust Tree objects (no string roundtrip).
    fn subtrees(&self) -> Vec<Py<Self>> {
        Python::try_attach(|py| {
            let mut result = Vec::new();
            self.collect_subtrees_py(py, &mut result);
            result
        })
        .expect("Python GIL not available")
    }

    fn pprint(&self) -> String {
        format!("{self}")
    }

    /// Append a child — accepts a `str` (leaf) or `Tree` (subtree).
    /// Matches NLTK's `tree.append(child)`.
    fn append(&mut self, child: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(tree) = child.extract::<Self>() {
            self.children.push(TreeNode::Subtree(tree));
        } else if let Ok(s) = child.extract::<String>() {
            self.children.push(TreeNode::Leaf(s));
        } else {
            let s: String = child.str()?.to_string();
            self.children.push(TreeNode::Leaf(s));
        }
        Ok(())
    }
}

impl fmt::Debug for Tree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tree(\"{}\"", self.label)?;
        for child in &self.children {
            write!(f, " {child:?}")?;
        }
        write!(f, ")")
    }
}

impl fmt::Display for Tree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}", self.label)?;
        for child in &self.children {
            write!(f, " {child}")?;
        }
        write!(f, ")")
    }
}

impl fmt::Display for TreeNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Leaf(s) => write!(f, "{s}"),
            Self::Subtree(t) => write!(f, "{t}"),
        }
    }
}

impl fmt::Debug for TreeNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Leaf(s) => write!(f, "\"{s}\""),
            Self::Subtree(t) => write!(f, "{t:?}"),
        }
    }
}

impl Tree {
    fn children_repr(&self) -> String {
        let parts: Vec<String> = self.children.iter().map(|c| format!("{c:?}")).collect();
        format!("[{}]", parts.join(", "))
    }

    fn collect_leaves(&self, result: &mut Vec<String>) {
        for child in &self.children {
            match child {
                TreeNode::Leaf(s) => result.push(s.clone()),
                TreeNode::Subtree(t) => t.collect_leaves(result),
            }
        }
    }

    fn collect_treepositions(&self, path: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
        for (i, child) in self.children.iter().enumerate() {
            path.push(i);
            match child {
                TreeNode::Leaf(_) => {
                    result.push(path.clone());
                }
                TreeNode::Subtree(t) => t.collect_treepositions(path, result),
            }
            path.pop();
        }
    }

    fn collect_productions(&self, result: &mut Vec<String>) {
        let rhs: Vec<String> = self
            .children
            .iter()
            .map(|c| match c {
                TreeNode::Leaf(s) => format!("'{s}'"),
                TreeNode::Subtree(t) => t.label.clone(),
            })
            .collect();
        if !rhs.is_empty() {
            result.push(format!("{} -> {}", self.label, rhs.join(" ")));
        }
        for child in &self.children {
            if let TreeNode::Subtree(t) = child {
                t.collect_productions(result);
            }
        }
    }

    fn collect_subtrees_py(&self, py: Python<'_>, result: &mut Vec<Py<Self>>) {
        if let Ok(obj) = Py::new(py, self.clone()) {
            result.push(obj);
        }
        for child in &self.children {
            if let TreeNode::Subtree(t) = child {
                t.collect_subtrees_py(py, result);
            }
        }
    }

    fn parse_brackets(s: &str, pos: usize) -> Result<(Self, usize), String> {
        let chars: Vec<char> = s.chars().collect();
        if pos >= chars.len() || chars[pos] != '(' {
            return Err("Expected '('".to_string());
        }

        let mut label_end = pos + 1;
        while label_end < chars.len() && chars[label_end] != ' ' && chars[label_end] != ')' {
            label_end += 1;
        }
        let label: String = chars[pos + 1..label_end].iter().collect();
        if label.is_empty() {
            return Err("Empty label".to_string());
        }

        let mut children = Vec::new();
        let mut i = label_end;

        while i < chars.len() {
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            if i >= chars.len() {
                break;
            }

            if chars[i] == ')' {
                return Ok((Self { label: label.trim().to_string(), children }, i + 1));
            } else if chars[i] == '(' {
                let (subtree, new_pos) = Self::parse_brackets(s, i)?;
                children.push(TreeNode::Subtree(subtree));
                i = new_pos;
            } else {
                let mut word_end = i;
                while word_end < chars.len()
                    && !chars[word_end].is_whitespace()
                    && chars[word_end] != ')'
                {
                    word_end += 1;
                }
                let word: String = chars[i..word_end].iter().collect();
                if !word.is_empty() {
                    children.push(TreeNode::Leaf(word));
                }
                i = word_end;
            }
        }

        Err("Unclosed bracket".to_string())
    }
}

// Registration

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tree>()?;
    Ok(())
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tree() -> Tree {
        Tree {
            label: "S".to_string(),
            children: vec![
                TreeNode::Leaf("The".to_string()),
                TreeNode::Subtree(Tree {
                    label: "NP".to_string(),
                    children: vec![TreeNode::Leaf("cat".to_string())],
                }),
            ],
        }
    }

    #[test]
    fn test_label() {
        assert_eq!(sample_tree().label(), "S");
    }

    #[test]
    fn test_len_counts_leaves() {
        assert_eq!(sample_tree().__len__(), 2);
    }

    #[test]
    fn test_leaves() {
        assert_eq!(sample_tree().leaves(), vec!["The", "cat"]);
    }

    #[test]
    fn test_height() {
        assert_eq!(sample_tree().height(), 3);
    }

    #[test]
    fn test_productions() {
        let prods = sample_tree().productions();
        assert!(prods.iter().any(|p| p.contains("S ->")));
        assert!(prods.iter().any(|p| p.contains("NP ->")));
    }

    #[test]
    fn test_from_string() {
        let tree = Tree::from_string("(S (NP The) (VP runs))").unwrap();
        assert_eq!(tree.label(), "S");
        assert_eq!(tree.leaves(), vec!["The", "runs"]);
    }

    #[test]
    fn test_from_string_nested() {
        let tree = Tree::from_string("(S (NP (Det The) (N cat)) (VP (V runs)))").unwrap();
        assert_eq!(tree.leaves(), vec!["The", "cat", "runs"]);
        assert!(tree.height() >= 4);
    }

    #[test]
    fn test_display() {
        let s = format!("{}", sample_tree());
        assert!(s.starts_with('('));
        assert!(s.contains('S'));
        assert!(s.contains("The"));
    }

    #[test]
    fn test_invalid() {
        assert!(Tree::from_string("not a tree").is_err());
    }

    #[test]
    fn test_leaf_treepositions() {
        let poses = sample_tree().leaf_treepositions();
        assert_eq!(poses.len(), 2);
    }

    #[test]
    fn test_subtrees_returns_py_objects() {
        pyo3::Python::initialize();
        let subs = sample_tree().subtrees();
        // Should have at least S and NP subtrees
        assert!(subs.len() >= 2);
    }

    #[test]
    fn test_append_string_and_tree() {
        pyo3::Python::initialize();
        pyo3::Python::try_attach(|py| {
            let mut tree = Tree::new("S", None);
            let child_str = "hello".to_string();
            tree.append(&child_str.into_pyobject(py).unwrap().into_any())?;
            let child_tree = Tree::new("NP", Some(vec!["world".to_string()]));
            let child_tree_py = Py::new(py, child_tree)?;
            tree.append(child_tree_py.bind(py))?;
            // Leaves should be: "hello", "world"
            assert_eq!(tree.leaves(), vec!["hello", "world"]);
            Ok::<_, PyErr>(())
        })
        .unwrap()
        .unwrap();
    }
}
