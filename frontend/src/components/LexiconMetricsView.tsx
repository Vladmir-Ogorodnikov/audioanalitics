import React, { useEffect, useState } from "react";
import { api } from "../api";
import type { LexiconResponse, MetricsEvalResponse } from "../types";

interface Props {
  onNotify: (msg: string) => void;
  onError: (err: string) => void;
}

const PRESET_REF =
  "Здравствуйте, коллеги! В худшем случае художник проверяет тригонометрию, гомоскедастичность и метод наименьших квадратов.";
const PRESET_HYP =
  "здравствуйте коллеги в худшем случае художник блядь проверяет триганометрия гомоскедастичнасть и метод наименьших квадратав";

export const LexiconMetricsView: React.FC<Props> = ({ onNotify, onError }) => {
  const [lexicon, setLexicon] = useState<LexiconResponse | null>(null);
  const [termsText, setTermsText] = useState("");
  const [badwordsText, setBadwordsText] = useState("");
  const [saving, setSaving] = useState(false);

  const [reference, setReference] = useState(PRESET_REF);
  const [hypothesis, setHypothesis] = useState(PRESET_HYP);
  const [applyPostproc, setApplyPostproc] = useState(true);
  const [caseSensitive, setCaseSensitive] = useState(false);
  const [evalRes, setEvalRes] = useState<MetricsEvalResponse | null>(null);

  const loadData = async () => {
    try {
      const data = await api.getLexicon();
      setLexicon(data);
      setTermsText(data.terms_text);
      setBadwordsText(data.badwords_text);
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    }
  };

  const runEval = async (
    refVal = reference,
    hypVal = hypothesis,
    postproc = applyPostproc,
    cs = caseSensitive
  ) => {
    try {
      const res = await api.evaluateMetrics({
        reference: refVal,
        hypothesis: hypVal,
        apply_postprocessing_preview: postproc,
        case_sensitive: cs,
      });
      setEvalRes(res);
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    }
  };

  useEffect(() => {
    loadData();
    runEval(PRESET_REF, PRESET_HYP, true, false);
  }, []);

  const handleSaveLexicon = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    try {
      const res = await api.saveLexicon(termsText, badwordsText);
      onNotify(
        `Словари сохранены: академических терминов — ${res.terms_count}, активных корней [НБЗ] — ${res.active_profanity_roots_count}.`
      );
      await loadData();
      await runEval();
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.2rem" }}>
      {/* Интерактивный стенд проверки исправлений Спринта 1 */}
      <div className="card">
        <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "center", gap: "0.5rem", marginBottom: "0.5rem" }}>
          <h2 style={{ margin: 0, fontSize: "1.14rem" }}>
            Стенд проверки метрик NFR-01 и исправлений Спринта 1 (WER / CER / Term F1 / [НБЗ])
          </h2>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={() => {
              setReference(PRESET_REF);
              setHypothesis(PRESET_HYP);
              setApplyPostproc(true);
              setCaseSensitive(false);
              runEval(PRESET_REF, PRESET_HYP, true, false);
            }}
          >
            Подставить контрольный пример
          </button>
        </div>
        <p style={{ margin: "0 0 0.9rem", fontSize: "0.86rem", color: "var(--text-muted)" }}>
          Проверьте прямо здесь все исправленные ошибки: (1) регистр и пунктуация больше не завышают WER/CER;
          (2) слова «худшем» и «художник» не заменяются ложно на [НБЗ], а реальный мат маркируется и в тексте, и в пословных чипах;
          (3) опечатки ASR в терминах исправляются по словарю и рассчитывается F1 по терминам (табл. 17 ТЗ).
        </p>

        <div className="grid-2" style={{ marginBottom: "0.85rem" }}>
          <div>
            <label className="label">Эталонная расшифровка (Reference):</label>
            <textarea
              rows={3}
              className="textarea"
              style={{ fontFamily: "inherit" }}
              value={reference}
              onChange={(e) => setReference(e.target.value)}
            />
          </div>
          <div>
            <label className="label">Гипотеза ASR (с опечатками, разным регистром и лексикой):</label>
            <textarea
              rows={3}
              className="textarea"
              style={{ fontFamily: "inherit" }}
              value={hypothesis}
              onChange={(e) => setHypothesis(e.target.value)}
            />
          </div>
        </div>

        <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "center", gap: "1rem", marginBottom: "1rem" }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "1.2rem", fontSize: "0.88rem" }}>
            <label style={{ display: "flex", alignItems: "center", gap: "0.4rem", cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={applyPostproc}
                onChange={(e) => {
                  setApplyPostproc(e.target.checked);
                  runEval(reference, hypothesis, e.target.checked, caseSensitive);
                }}
              />
              Включить постобработку опечаток по словарю терминов
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: "0.4rem", cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={caseSensitive}
                onChange={(e) => {
                  setCaseSensitive(e.target.checked);
                  runEval(reference, hypothesis, applyPostproc, e.target.checked);
                }}
              />
              Учитывать регистр (старый режим до исправления)
            </label>
          </div>

          <button type="button" className="btn btn-primary" onClick={() => runEval()}>
            Пересчитать метрики
          </button>
        </div>

        {evalRes && (
          <div>
            <div className="grid-4" style={{ marginBottom: "1rem" }}>
              <div className="stat-card">
                <div className="stat-label">WER (порог ≤ 12%)</div>
                <div className="stat-value">{evalRes.wer.percent}%</div>
                <span className={`pill ${evalRes.wer.passes_threshold_12pct ? "pill-green" : "pill-red"}`}>
                  {evalRes.wer.passes_threshold_12pct ? "Соответствует ТЗ" : "Превышает порог"}
                </span>
              </div>

              <div className="stat-card">
                <div className="stat-label">CER (порог ≤ 5%)</div>
                <div className="stat-value">{evalRes.cer.percent}%</div>
                <span className={`pill ${evalRes.cer.passes_threshold_5pct ? "pill-green" : "pill-red"}`}>
                  {evalRes.cer.passes_threshold_5pct ? "Соответствует ТЗ" : "Превышает порог"}
                </span>
              </div>

              <div className="stat-card">
                <div className="stat-label">Полнота терминов (≥ 85%)</div>
                <div className="stat-value">{(evalRes.terms.recall * 100).toFixed(1)}%</div>
                <span className={`pill ${evalRes.terms.passes_recall_85pct ? "pill-green" : "pill-red"}`}>
                  Найдено {evalRes.terms.true_positives} из {evalRes.terms.ref_occurrences}
                </span>
              </div>

              <div className="stat-card">
                <div className="stat-label">F1 по терминам (≥ 0.88)</div>
                <div className="stat-value">{evalRes.terms.f1.toFixed(3)}</div>
                <span className={`pill ${evalRes.terms.passes_f1_088 ? "pill-green" : "pill-red"}`}>
                  Precision {(evalRes.terms.precision * 100).toFixed(0)}%
                </span>
              </div>
            </div>

            <div style={{ padding: "0.85rem 1rem", background: "var(--surface-alt)", borderRadius: "10px", border: "1px solid var(--border)" }}>
              <div style={{ marginBottom: "0.45rem", fontSize: "0.9rem" }}>
                <strong>Текст после фильтра [НБЗ] и постобработки терминов:</strong>{" "}
                <span>{evalRes.processed_hypothesis}</span>
              </div>
              <div style={{ marginBottom: "0.45rem", fontSize: "0.86rem" }}>
                <strong>Пословные чипы плеера (Segment.words):</strong>{" "}
                {evalRes.word_level_preview.map((w, idx) => (
                  <span
                    key={idx}
                    className={`word-chip ${w.includes("[НБЗ]") ? "nbz" : ""}`}
                    style={{ cursor: "default", border: "1px solid var(--border)", marginRight: "4px" }}
                  >
                    {w}
                  </span>
                ))}
              </div>
              {evalRes.term_corrections.length > 0 && (
                <div style={{ fontSize: "0.84rem", color: "var(--success)" }}>
                  <strong>Исправлены опечатки по словарю:</strong>{" "}
                  {evalRes.term_corrections.map((c) => `${c.from} → ${c.to}`).join("; ")}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Редактор пользовательских словарей п. 4.10 */}
      <form onSubmit={handleSaveLexicon} className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
          <h2 style={{ margin: 0, fontSize: "1.12rem" }}>
            Редактор словарей лингвистического обеспечения (п. 4.10 ТЗ)
          </h2>
          <button type="submit" className="btn btn-primary" disabled={saving}>
            {saving ? "Сохранение..." : "Сохранить словари"}
          </button>
        </div>

        <div className="grid-2">
          <div>
            <label className="label">
              Академический словарь терминов (`terms.txt`, сейчас: {lexicon?.terms_count ?? 0})
            </label>
            <textarea
              rows={12}
              className="textarea"
              value={termsText}
              onChange={(e) => setTermsText(e.target.value)}
            />
          </div>

          <div>
            <label className="label">
              Дополнительные корни ненормативной лексики [НБЗ] (`badwords.txt`, всего активных корней:{" "}
              {lexicon?.active_profanity_roots_count ?? 0})
            </label>
            <textarea
              rows={8}
              className="textarea"
              value={badwordsText}
              onChange={(e) => setBadwordsText(e.target.value)}
            />
            {lexicon && (
              <div style={{ marginTop: "0.6rem", fontSize: "0.8rem", color: "var(--text-muted)" }}>
                <strong>Защищённые префиксы нейтральных слов (белый список):</strong>{" "}
                <code>{lexicon.safe_word_prefixes.join(", ")}</code>
              </div>
            )}
          </div>
        </div>
      </form>
    </div>
  );
};
