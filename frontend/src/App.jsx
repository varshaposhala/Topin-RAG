import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import { fetchCatalogs, searchQuestions } from "./api";

const EXAMPLES = [
  "give 10 python coding questions",
  "GRIT_CT_L1_POOL_1 questions",
  "show sql mcqs",
  "advanced git questions",
];

const PAGE_SIZE = 10;

function downloadCsv(csvText) {
  const blob = new Blob([csvText], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "topin_questions.csv";
  a.click();
  URL.revokeObjectURL(url);
}

export default function App() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [catalogs, setCatalogs] = useState(null);
  const [selection, setSelection] = useState(null);
  const [form, setForm] = useState({});
  const [result, setResult] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [page, setPage] = useState(1);

  useEffect(() => {
    fetchCatalogs()
      .then(setCatalogs)
      .catch(() => {
        /* catalogs also arrive with needs_selection */
      });
  }, []);

  const questions = result?.questions || [];
  const totalPages = Math.max(1, Math.ceil(questions.length / PAGE_SIZE));
  const visible = useMemo(() => {
    const start = (page - 1) * PAGE_SIZE;
    return questions.slice(start, start + PAGE_SIZE);
  }, [questions, page]);

  async function runSearch(nextQuery, extras = {}) {
    const q = (nextQuery ?? query).trim();
    if (!q) return;
    setLoading(true);
    setError("");
    try {
      const data = await searchQuestions({
        query: q,
        session_id: sessionId,
        ...extras,
      });
      if (data.type === "needs_selection") {
        setSelection(data);
        setResult(null);
        setForm({});
      } else if (data.type === "empty") {
        setSelection(null);
        setResult(data);
      } else {
        setSelection(null);
        setResult(data);
        setSessionId(data.session_id || null);
        setPage(1);
      }
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(e) {
    e.preventDefault();
    runSearch(query);
  }

  function submitFilters(e) {
    e.preventDefault();
    const missing = selection?.missing || [];
    for (const field of missing) {
      const key =
        field === "count"
          ? "count_choice"
          : field === "question_type"
            ? "question_type"
            : field;
      if (!form[key] || form[key] === "any") {
        if (field === "difficulty" && form.difficulty === "any") continue;
        if (field === "difficulty" && !form.difficulty) {
          setError("Please complete all filters");
          return;
        }
        if (field !== "difficulty") {
          setError("Please complete all filters");
          return;
        }
      }
    }
    runSearch(query, {
      selections: form,
      partial_intent: selection?.partial_intent,
    });
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <h1 className="brand">Topin</h1>
        <p className="hero-line">
          Search the question bank in plain language — by topic, tag, difficulty, or question ID.
        </p>
      </header>

      <form className="search-panel" onSubmit={onSubmit}>
        <div className="search-row">
          <input
            className="search-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder='Try “give 10 python coding questions” or a GRIT tag'
            aria-label="Search query"
          />
          <button className="btn btn-primary" type="submit" disabled={loading}>
            {loading ? "Searching…" : "Search"}
          </button>
        </div>
        <div className="chips">
          {EXAMPLES.map((example) => (
            <button
              key={example}
              type="button"
              className="chip"
              onClick={() => {
                setQuery(example);
                runSearch(example);
              }}
            >
              {example}
            </button>
          ))}
        </div>
      </form>

      {error ? <div className="error">{error}</div> : null}
      {loading ? <p className="status">Looking through Topin questions…</p> : null}

      {selection?.type === "needs_selection" ? (
        <section className="panel">
          <h2>Refine your search</h2>
          <p className="muted">{selection.message}</p>
          <form className="filters" onSubmit={submitFilters}>
            {(selection.missing || []).includes("topic") ? (
              <label>
                Topic
                <select
                  value={form.topic || ""}
                  onChange={(e) => setForm((f) => ({ ...f, topic: e.target.value }))}
                >
                  <option value="">Select topic</option>
                  {(selection.catalogs?.topics || catalogs?.topics || []).map((t) => (
                    <option key={t.value} value={t.value}>
                      {t.label}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}

            {(selection.missing || []).includes("subject") ? (
              <label>
                Subject
                <select
                  value={form.subject || ""}
                  onChange={(e) => setForm((f) => ({ ...f, subject: e.target.value }))}
                >
                  <option value="">Select subject</option>
                  {(selection.catalogs?.subjects || catalogs?.subjects || []).map((s) => (
                    <option key={s.value} value={s.value}>
                      {s.label}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}

            {(selection.missing || []).includes("question_type") ? (
              <label>
                Question type
                <select
                  value={form.question_type || ""}
                  onChange={(e) => setForm((f) => ({ ...f, question_type: e.target.value }))}
                >
                  <option value="">Select type</option>
                  {(selection.catalogs?.question_types || catalogs?.question_types || []).map(
                    (t) => (
                      <option key={t.value} value={t.value}>
                        {t.label}
                      </option>
                    )
                  )}
                </select>
              </label>
            ) : null}

            {(selection.missing || []).includes("count") ? (
              <label>
                How many
                <select
                  value={form.count_choice || ""}
                  onChange={(e) => setForm((f) => ({ ...f, count_choice: e.target.value }))}
                >
                  <option value="">Select count</option>
                  {(selection.catalogs?.counts || catalogs?.counts || []).map((c) => (
                    <option key={c.value} value={c.value}>
                      {c.label}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}

            {(selection.missing || []).includes("difficulty") ? (
              <label>
                Difficulty
                <select
                  value={form.difficulty || ""}
                  onChange={(e) => setForm((f) => ({ ...f, difficulty: e.target.value }))}
                >
                  <option value="">Select difficulty</option>
                  {(selection.catalogs?.difficulties || catalogs?.difficulties || []).map((d) => (
                    <option key={d.value} value={d.value}>
                      {d.label}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}

            <div style={{ display: "flex", alignItems: "end" }}>
              <button className="btn btn-primary" type="submit" disabled={loading}>
                Apply filters
              </button>
            </div>
          </form>
        </section>
      ) : null}

      {result?.type === "empty" ? (
        <section className="panel">
          <h2>No matches</h2>
          <p className="muted">{result.message}</p>
        </section>
      ) : null}

      {result?.type === "results" ? (
        <section>
          <div className="results-head">
            <div>
              <h2>{questions.length} question{questions.length === 1 ? "" : "s"}</h2>
              <p className="muted">{result.label}</p>
            </div>
            {result.csv ? (
              <button className="btn btn-ghost" type="button" onClick={() => downloadCsv(result.csv)}>
                Download CSV
              </button>
            ) : null}
          </div>

          <div className="question-list">
            {visible.map((q) => (
              <article className="question" key={`${q.question_id}-${q.index}`}>
                <div className="question-meta">
                  <span className="badge">Q{q.index}</span>
                  {q.is_coding ? <span className="badge coral">Coding</span> : null}
                  {q.difficulty ? <span className="badge">{q.difficulty}</span> : null}
                  {q.topic ? <span className="badge">{q.topic}</span> : null}
                  {q.subtopic ? <span className="badge">{q.subtopic}</span> : null}
                </div>
                <div className="question-body">
                  <ReactMarkdown>{q.question_text || ""}</ReactMarkdown>
                </div>
                {q.options?.length ? (
                  <div className="options">
                    {q.options.map((opt) => (
                      <div
                        key={`${q.question_id}-${opt.label}`}
                        className={`option${opt.is_correct ? " correct" : ""}`}
                      >
                        <strong>{opt.label}.</strong> {opt.text}
                      </div>
                    ))}
                  </div>
                ) : null}
                {q.tags?.length ? (
                  <div className="tags">
                    {q.tags.map((tag) => (
                      <span
                        key={tag}
                        className={`tag${(q.matched_tags || []).includes(tag) ? " matched" : ""}`}
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                ) : null}
                {q.question_id ? <div className="qid">ID: {q.question_id}</div> : null}
              </article>
            ))}
          </div>

          {totalPages > 1 ? (
            <div className="pager">
              <button
                className="btn btn-ghost"
                type="button"
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
              >
                Previous
              </button>
              <span className="muted">
                Page {page} of {totalPages}
              </span>
              <button
                className="btn btn-ghost"
                type="button"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              >
                Next
              </button>
            </div>
          ) : null}
        </section>
      ) : null}
    </div>
  );
}
