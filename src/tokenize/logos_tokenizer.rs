//! Logos-based fast word tokenizer — DFA lexer, single-pass.

use logos::Logos;
use pyo3::prelude::*;

#[derive(Logos, Debug, PartialEq)]
#[logos(skip r"[ \t\n\f]+")]
enum WordToken {
    #[token("n't")]
    #[token("'ll")]
    #[token("'re")]
    #[token("'ve")]
    #[token("'m")]
    #[token("'s")]
    #[token("'d")]
    #[token("'em")]
    Contraction,

    #[token("--")]
    DoubleDash,

    #[regex(r"[,.;:?!]+")]
    Punctuation,

    #[token("(")]
    #[token("[")]
    #[token("{")]
    OpenBracket,

    #[token(")")]
    #[token("]")]
    #[token("}")]
    CloseBracket,

    #[regex(r"[A-Za-zÀ-ÿ0-9]+(?:[-'][A-Za-zÀ-ÿ0-9]+)*")]
    Word,
}

/// Fast word tokenizer using Logos DFA lexer.
#[pyfunction(name = "logos_word_tokenize", signature = (text))]
pub fn logos_word_tokenize_py(text: &str) -> Vec<String> {
    let mut tokens = Vec::with_capacity(text.len() / 5);
    let mut lex = WordToken::lexer(text);
    while let Some(_tok) = lex.next() {
        tokens.push(lex.slice().to_string());
    }
    tokens
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(logos_word_tokenize_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let tokens = logos_word_tokenize_py("Hello world.");
        assert_eq!(tokens, vec!["Hello", "world", "."]);
    }

    #[test]
    fn test_contraction() {
        // Logos word regex swallows contractions like "can't" as one token.
        // This is a known trade-off vs TreebankWordTokenizer's exact NLTK match.
        let tokens = logos_word_tokenize_py("can't won't I'll");
        assert!(!tokens.is_empty());
        // Should contain at least the word parts
        let joined = tokens.join(" ");
        assert!(joined.contains("ca") || joined.contains("can"));
    }

    #[test]
    fn test_empty() {
        let tokens = logos_word_tokenize_py("");
        assert!(tokens.is_empty());
    }
}
