import ts from "typescript";

// Finds English a user would read that did not come from t(). Parses the TSX
// with the TypeScript compiler rather than a regex, so JSX text split across
// lines, nested ternaries and template literals are seen for what they are.
//
// Sinks (where a literal becomes visible):
//   * JSX text between tags                          <th>Provider</th>
//   * display attributes: title, aria-label, alt, placeholder, label, ...
//   * a literal rendered as a JSX child expression   {ok ? "Saved" : "Failed"}
// Not copy, and skipped: t("key", ...) arguments, comparison operands
// (`status === "running"`), string literals passed straight to any call
// (`filterBtn("open", ...)`, `setTab("settings")`), and statement blocks inside
// an expression (`{(() => { const className = "px-2 ..."; })()}`).
//
// KNOWN LIMIT: a string built in ordinary code and rendered later through a
// variable (`const tip = \`cost: $${x}\`` ... `title={tip}`) is not a sink here.
// Those were swept by hand when this gate landed; the gate keeps new JSX copy out.

export type Hit = { line: number; text: string; kind: string };

const DISPLAY_ATTR =
  /^(title|alt|placeholder|label|aria-label|aria-description|aria-valuetext|caption|hint|message|tooltip|description|heading|subtitle|emptyText|helperText)$/i;
const WORDS = /[A-Za-z]{2,}/;
// Tailwind-ish class lists and bare identifiers are not prose.
const CLASS_LIST = /^[a-z0-9:/[\]._%#-]+(\s+[a-z0-9:/[\]._%#-]+)*$/;
const IDENTIFIER = /^[a-z][a-z0-9_]*$/;

function literalText(node: ts.Node): string | null {
  if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) return node.text;
  if (ts.isTemplateExpression(node)) {
    return [node.head.text, ...node.templateSpans.map((s) => s.literal.text)].join("{}");
  }
  return null;
}

const COMPARISON = new Set([
  ts.SyntaxKind.EqualsEqualsEqualsToken,
  ts.SyntaxKind.ExclamationEqualsEqualsToken,
  ts.SyntaxKind.EqualsEqualsToken,
  ts.SyntaxKind.ExclamationEqualsToken,
]);

function literalsIn(expression: ts.Node, sourceFile: ts.SourceFile): ts.Node[] {
  const out: ts.Node[] = [];
  const visit = (node: ts.Node): void => {
    if (ts.isBlock(node)) return;
    if (ts.isJsxElement(node) || ts.isJsxSelfClosingElement(node) || ts.isJsxFragment(node)) return;
    if (ts.isCallExpression(node)) {
      const callee = node.expression.getText(sourceFile);
      if (/(^|\.)t$/.test(callee)) return;
      ts.forEachChild(node.expression, visit);
      for (const arg of node.arguments) if (literalText(arg) === null) visit(arg);
      return;
    }
    if (ts.isBinaryExpression(node) && COMPARISON.has(node.operatorToken.kind)) return;
    const text = literalText(node);
    if (text !== null) {
      if (WORDS.test(text)) out.push(node);
      return;
    }
    ts.forEachChild(node, visit);
  };
  visit(expression);
  return out;
}

export function scanSource(fileName: string, source: string): Hit[] {
  const sf = ts.createSourceFile(fileName, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const hits: Hit[] = [];
  const add = (node: ts.Node, text: string, kind: string) => {
    hits.push({ line: sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1, text, kind });
  };
  const walk = (node: ts.Node): void => {
    if (ts.isJsxText(node)) {
      const text = node.text.replace(/\s+/g, " ").trim();
      if (WORDS.test(text)) add(node, text, "text");
    } else if (ts.isJsxAttribute(node) && node.initializer && DISPLAY_ATTR.test(node.name.getText(sf))) {
      const init = node.initializer;
      const kind = `@${node.name.getText(sf)}`;
      if (ts.isStringLiteral(init)) {
        if (WORDS.test(init.text)) add(init, init.text, kind);
      } else if (ts.isJsxExpression(init) && init.expression) {
        for (const lit of literalsIn(init.expression, sf)) add(lit, literalText(lit)!, kind);
      }
    } else if (
      ts.isJsxExpression(node) &&
      node.expression &&
      node.parent &&
      (ts.isJsxElement(node.parent) || ts.isJsxFragment(node.parent))
    ) {
      for (const lit of literalsIn(node.expression, sf)) {
        const text = literalText(lit)!;
        if (CLASS_LIST.test(text) && (IDENTIFIER.test(text) || /[-:/\d]/.test(text))) continue;
        add(lit, text, "child");
      }
    }
    ts.forEachChild(node, walk);
  };
  walk(sf);
  return hits;
}
