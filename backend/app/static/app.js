/*
  Console behaviour.

  Nothing here invents a number. Every value rendered comes from one of the six
  /api endpoints, and the evaluation figures come from the JSON `make eval`
  writes alongside EVALUATION.md. If an endpoint is unavailable the panel says
  so rather than falling back to a plausible-looking constant -- a page that
  degrades into fiction is worse than one that degrades into a blank.
*/

const rupees = new Intl.NumberFormat("en-IN", {
  style: "currency", currency: "INR", maximumFractionDigits: 0,
});
const money = (n) => (n == null ? "—" : rupees.format(n));
const count = (n) => (n == null ? "—" : new Intl.NumberFormat("en-IN").format(n));
const pct = (x, digits = 0) => (x == null ? "—" : `${x >= 0 ? "+" : ""}${(x * 100).toFixed(digits)}%`);

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, html) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (html != null) node.innerHTML = html;
  return node;
};
const esc = (s) =>
  String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

async function get(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json();
}

function fail(sel, err) {
  const node = $(sel);
  if (node) node.innerHTML = `<p class="err">Unavailable — ${esc(err.message)}</p>`;
}

/* ── reveal on scroll ─────────────────────────────────────────────── */

const seen = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entry.target.classList.add("in");
      seen.unobserve(entry.target);
    }
  },
  { threshold: 0.15, rootMargin: "0px 0px -8% 0px" },
);
const watch = (root = document) =>
  root.querySelectorAll(".reveal:not(.in), .markline:not(.in), .step:not(.in)").forEach((n) => seen.observe(n));

/* ── section nav ──────────────────────────────────────────────────── */

function nav() {
  const links = [...document.querySelectorAll(".nav a")];
  const sections = links
    .map((a) => ({ link: a, section: document.querySelector(a.getAttribute("href")) }))
    .filter((x) => x.section);

  const active = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        const match = sections.find((s) => s.section === entry.target);
        if (!match) continue;
        if (entry.isIntersecting) {
          links.forEach((l) => l.classList.remove("on"));
          match.link.classList.add("on");
        }
      }
    },
    { threshold: 0.25, rootMargin: "-25% 0px -55% 0px" },
  );
  sections.forEach((s) => active.observe(s.section));
}

/* ── act 1: the real recovery ─────────────────────────────────────── */

/* Each event describes itself from its own payload, so this reads any case's
   ledger rather than the one case it was written against. */
const NARRATE = {
  "case.opened": (p, c) => [
    `A ${money(c.amount_rupees)} payment failed`,
    [p.error_reason && `error_reason <strong>${esc(p.error_reason)}</strong>`,
     p.error_source && `source <strong>${esc(p.error_source)}</strong>`,
     p.error_step && `step <strong>${esc(p.error_step)}</strong>`]
      .filter(Boolean).join(" · "),
  ],
  "case.diagnosed": (p) => [
    `Classified as <strong>${esc(p.cause)}</strong>`,
    `${esc(p.method)} · confidence ${Number(p.confidence).toFixed(2)}` +
      (p.matched_on ? ` · matched on <span class="mono">${esc(p.matched_on)}</span>` : "") +
      (p.model ? ` · ${esc(p.model)}` : " · no model consulted"),
  ],
  "case.diagnosis_deferred": () => ["Deferred for classification", "Deterministic lookup did not resolve it"],
  "action.claimed": (p) => [
    `Chose <strong>${esc(String(p.action_type).replace(/_/g, " "))}</strong>`,
    [p.predicted_uplift != null && `predicted uplift ${(p.predicted_uplift * 100).toFixed(1)}%`,
     p.expected_value_paise != null && `expected value ${money(p.expected_value_paise / 100)}`,
     p.rule && `rule <span class="mono">${esc(p.rule)}</span>`]
      .filter(Boolean).join(" · "),
  ],
  "case.stopped": (p) => [`Held back — <strong>${esc(p.rule)}</strong>`, esc(p.reason || "")],
  "action.executed": (p) => [
    "Recovery action sent",
    esc(p.detail || "") + (p.short_url ? ` · <span class="mono">${esc(p.short_url)}</span>` : ""),
  ],
  "case.suppressed": (p) => [
    `Refused — <strong>${esc(String(p.rule || "suppressed").replace(/_/g, " "))}</strong>`,
    esc(p.reason || "no action has positive expected value here"),
  ],
  "case.action_deferred": (p) => [
    "Decided, and deliberately held",
    [p.rule && `rule <span class="mono">${esc(p.rule)}</span>`,
     p.send_at && `sending ${esc(String(p.send_at).replace("T", " ").slice(0, 16))}`,
     p.reason && esc(p.reason)].filter(Boolean).join(" · "),
  ],
  "case.recovered": (p, c) => [
    `<strong>${money(c.amount_rupees)} paid</strong>`,
    (p.matched_by ? `attributed by <span class="mono">${esc(p.matched_by)}</span>` : "") +
      (p.recovering_payment_id ? ` · <span class="mono">${esc(p.recovering_payment_id)}</span>` : ""),
  ],
};

function narrate(event, caseRow) {
  const fn = NARRATE[event.event];
  if (fn) return fn(event.payload || {}, caseRow);
  return [esc(event.event), ""];
}

async function story() {
  let data;
  try {
    data = await get("/api/evidence");
  } catch (err) {
    fail("#story", err);
    return;
  }
  const c = data.case;
  const events = data.ledger || [];
  $("#hero-amount").textContent = money(c.amount_rupees);

  const rail = $("#story");
  rail.innerHTML = "";
  const t0 = events.length ? new Date(events[0].at).getTime() : 0;

  events.forEach((event) => {
    const [what, detail] = narrate(event, c);
    const secs = Math.round((new Date(event.at).getTime() - t0) / 1000);
    const step = el("div", "step reveal" + (event.event === "case.recovered" ? " done" : ""));
    step.innerHTML =
      `<div class="when">+${secs}s</div>` +
      `<div class="what">${what}<span class="who">${esc(event.actor)}</span></div>` +
      (detail ? `<div class="small muted">${detail}</div>` : "");
    rail.appendChild(step);
  });

  const total = events.length
    ? Math.round((new Date(events.at(-1).at).getTime() - t0) / 1000)
    : 0;
  // Computed, never typed. This headline read "Forty-eight seconds" while the
  // ledger directly below it said 78 -- on the one section of the page that
  // claims nothing here is written for display.
  $("#story-headline").textContent =
    `One payment failed. ${total} seconds later it was paid.`;

  $("#story-note").innerHTML =
    `<p style="margin:0">Payment <span class="mono">${esc(c.external_ref)}</span> · ` +
    `${events.length} ledger entries · ${total} seconds from failure to payment. ` +
    `Razorpay test mode, cross-checked against their dashboard: the recovery link reads <strong>paid</strong>.</p>`;

  taxonomy(data);
  watch(rail);
  requestAnimationFrame(() => rail.style.setProperty("--fill", "100%"));
}

function taxonomy(data) {
  const opened = (data.ledger || []).find((e) => e.event === "case.opened");
  const diagnosed = (data.ledger || []).find((e) => e.event === "case.diagnosed");
  const claimed = (data.ledger || []).find((e) => e.event === "action.claimed");
  const p = opened ? opened.payload : {};

  $("#tax-fields").innerHTML = [
    ["error_code", p.error_code],
    ["error_source", p.error_source],
    ["error_step", p.error_step],
    ["error_reason", p.error_reason],
  ]
    .map(([k, v]) => `<dt>${k}</dt><dd>${esc(v ?? "—")}</dd>`)
    .join("");

  const d = diagnosed ? diagnosed.payload : {};
  $("#tax-cause").innerHTML =
    `<p style="font-size:22px;margin:0 0 8px;color:var(--ink)">${esc(d.cause ?? "—")}</p>` +
    `<span class="tag blue">${esc(d.method ?? "—")}</span>` +
    `<p class="small" style="margin-top:14px">Matched on <span class="mono">${esc(d.matched_on ?? "—")}</span> ` +
    `at confidence ${d.confidence != null ? Number(d.confidence).toFixed(2) : "—"}.</p>`;

  const a = claimed ? claimed.payload : {};
  $("#tax-action").innerHTML =
    `<p style="font-size:22px;margin:0 0 8px;color:var(--ink)">${esc(String(a.action_type ?? "—").replace(/_/g, " "))}</p>` +
    `<span class="tag green">not a retry</span>` +
    `<p class="small" style="margin-top:14px">Expected value ${money(a.expected_value_paise != null ? a.expected_value_paise / 100 : null)} ` +
    `at ${a.predicted_uplift != null ? (a.predicted_uplift * 100).toFixed(1) + "%" : "—"} predicted uplift.</p>`;
}

/* ── act 3: the levers ────────────────────────────────────────────── */

function bars(target, rows, heroVariant) {
  const node = $(target);
  node.innerHTML = "";
  const peak = Math.max(...rows.map((r) => r.realised_rupees), 1);

  rows.forEach((r, i) => {
    const isBase = r.lift === 0;
    const isHero = r.variant === heroVariant;
    const bar = el("div", `bar reveal${isBase ? " base" : ""}${isHero ? " hero-bar" : ""}`);
    // The fill width rides on the same reveal observer as everything else,
    // rather than a second observer per bar. A separate one silently failed to
    // fire for elements created after first paint, and an empty bar chart looks
    // like a styling choice rather than a bug.
    bar.style.setProperty("--w", `${(r.realised_rupees / peak) * 100}%`);
    bar.dataset.d = String(Math.min(i + 1, 5));
    bar.innerHTML =
      `<div class="top"><span class="label">${esc(r.variant)}</span>` +
      `<span class="value">${money(r.realised_rupees)}` +
      `${isBase ? '<span class="muted"> · baseline</span>' : ` · <strong>${pct(r.lift)}</strong>`}</span></div>` +
      `<div class="track"><span class="fill"></span></div>`;
    node.appendChild(bar);
  });
  watch(node);
}

async function levers() {
  let facts;
  try {
    facts = await get("/api/facts");
  } catch (err) {
    fail("#lever-action", err);
    fail("#lever-ranking", err);
    return;
  }
  $("#w-cases").textContent = count(facts.test_cases);
  $("#w-budget").textContent = count(facts.contact_budget);

  const action = facts.levers.filter((l) => l.lever === "action");
  const ranking = facts.levers.filter((l) => l.lever === "ranking");
  const best = action.reduce((a, b) => (b.lift > a.lift ? b : a), action[0]);
  bars("#lever-action", action, best && best.variant);
  bars("#lever-ranking", ranking, null);
}

/* ── act 5: the console ───────────────────────────────────────────── */

async function overview() {
  let o;
  try {
    o = await get("/api/overview");
  } catch (err) {
    fail("#overview", err);
    return;
  }
  const q = o.quiet_hours;
  const cells = [
    ["Cases", count(o.cases_total), `${count(o.recovered_cases)} recovered`],
    ["Scheduled", count(o.scheduled), "decided, waiting for the right moment"],
    ["At risk", money(o.at_risk_rupees), "excludes recovered"],
    ["Recovered", money(o.recovered_rupees), "attributed to an action"],
    ["Awaiting approval", count(o.awaiting_approval), "a human decides"],
    [
      "Quiet hours",
      q.active ? "Active" : "Open",
      `${q.start_ist}:00–${q.end_ist}:00 IST · now ${q.now_ist}`,
    ],
  ];
  $("#overview").innerHTML = cells
    .map(([k, v, n]) => `<div class="cell stat"><div class="k">${k}</div><div class="v">${v}</div><div class="n">${n}</div></div>`)
    .join("");
}

function decisionRow(item) {
  const confident = item.cause_confidence != null && item.cause_confidence >= 0.7;
  const row = el("div", "row");
  row.dataset.actionId = item.action_id;
  row.innerHTML =
    `<div class="money">${money(item.amount_rupees)}</div>` +
    `<div class="body">` +
    `<div class="line1">${esc(String(item.cause ?? "unclassified").replace(/_/g, " "))} → ` +
    `<strong>${esc(String(item.action_type).replace(/_/g, " "))}</strong></div>` +
    `<div class="line2"><span class="mono">${esc(item.external_ref)}</span>` +
    (item.why_queued ? ` · ${esc(item.why_queued)}` : "") +
    (item.expected_value_rupees != null ? ` · EV ${money(item.expected_value_rupees)}` : "") +
    `</div></div>` +
    `<span class="tag ${confident ? "grey" : "amber"}">${confident ? esc(item.cause_method ?? "classified") : "low confidence"}</span>` +
    `<div class="acts">` +
    `<button class="primary" data-do="approve">Approve</button>` +
    `<button data-do="reject">Reject</button>` +
    `</div>`;
  return row;
}

async function queue() {
  let items;
  try {
    items = await get("/api/queue");
  } catch (err) {
    fail("#queue", err);
    return;
  }
  const node = $("#queue");
  node.innerHTML = "";
  if (!items.length) {
    node.innerHTML = '<p class="load">Nothing waiting. Run <span class="mono">make live-demo</span> to populate the queue.</p>';
    return;
  }
  items.forEach((item) => node.appendChild(decisionRow(item)));
}

/* Approval writes to the same endpoints the CLI uses. `live` stays false: this
   page must never be the thing that sends a real customer a message. */
async function decide(row, verb) {
  const buttons = [...row.querySelectorAll("button")];
  buttons.forEach((b) => (b.disabled = true));
  const body =
    verb === "approve"
      ? { approved_by: "console", execute_now: true, live: false }
      : { rejected_by: "console", reason: "rejected from the console" };
  try {
    const res = await fetch(`/actions/${row.dataset.actionId}/${verb}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`${res.status}`);
    row.querySelector(".acts").innerHTML =
      `<span class="tag ${verb === "approve" ? "green" : "red"}">${verb === "approve" ? "approved" : "rejected"}</span>`;
    row.style.opacity = "0.55";
    overview();
    ledger();
    rules();
    scheduled();
  } catch (err) {
    buttons.forEach((b) => (b.disabled = false));
    row.querySelector(".acts").insertAdjacentHTML("afterbegin", `<span class="tag red">${esc(err.message)}</span>`);
  }
}

document.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-do]");
  if (!button) return;
  decide(button.closest(".row"), button.dataset.do);
});

const RULE_COPY = {
  quiet_hours: "Send time moved out of 21:00–09:00 IST",
  terminal_status: "Case was already finished",
  human_rejected: "A reviewer said no; never re-proposed",
  action_already_pending: "Already awaiting a decision",
  max_actions_reached: "Three actions already taken",
  past_recovery_window: "Older than seven days",
  awaiting_diagnosis: "Cause not yet resolved",
  on_exception_list: "Cause unresolved; needs a human",
  risk_suppression: "A guardrail forbade contact entirely",
  ev_floor: "Not worth the contact it would cost",
};

/* Three outcomes, deliberately not merged into one "blocked" number.
   Quiet hours postpone a contact; they never cancel one, and counting a
   deferral as a refusal would overstate this panel in our own favour. */
const OUTCOME = {
  stopped: ["red", "never planned"],
  suppressed: ["amber", "planned, then declined"],
  deferred: ["blue", "postponed, not cancelled"],
};

const WAIT_COPY = {
  quiet_hours: "Would have landed inside quiet hours",
  contact_budget: "Waiting for the moment the cause implies",
  unrationed: "Waiting for the moment the cause implies",
};

async function scheduled() {
  let items;
  try {
    items = await get("/api/scheduled");
  } catch (err) {
    fail("#scheduled", err);
    return;
  }
  const node = $("#scheduled");
  if (!items.length) {
    node.innerHTML =
      '<p class="load">Nothing waiting to send — every decided contact is already due.</p>';
    return;
  }
  node.innerHTML = items
    .map((s) => {
      const hrs = s.due_in_hours;
      const when = hrs <= 0 ? "due now" : hrs < 1 ? `in ${Math.round(hrs * 60)} min` : `in ${hrs}h`;
      return (
        `<div class="row">` +
        `<div class="money">${money(s.amount_rupees)}</div>` +
        `<div class="body">` +
        `<div class="line1">${esc(String(s.cause ?? "").replace(/_/g, " "))} → ` +
        `<strong>${esc(String(s.action_type).replace(/_/g, " "))}</strong></div>` +
        `<div class="line2"><span class="mono">${esc(s.external_ref)}</span> · ` +
        `${esc(WAIT_COPY[s.rule] ?? "waiting")}</div></div>` +
        `<span class="tag blue">${esc(when)}</span></div>`
      );
    })
    .join("");
}

async function rules() {
  let items;
  try {
    items = await get("/api/rules");
  } catch (err) {
    fail("#rules", err);
    return;
  }
  const node = $("#rules");
  if (!items.length) {
    node.innerHTML = '<p class="load">Nothing has been held back yet.</p>';
    return;
  }
  node.innerHTML =
    '<div class="grid g3" style="border:0;border-radius:0">' +
    items
      .map((r) => {
        const [tone, meaning] = OUTCOME[r.outcome] ?? ["grey", r.outcome];
        return (
          `<div class="cell stat">` +
          `<div class="k">${esc(r.rule.replace(/_/g, " "))}</div>` +
          `<div class="v">${count(r.count)}</div>` +
          `<div style="margin:9px 0 7px"><span class="tag ${tone}">${esc(meaning)}</span></div>` +
          `<div class="n">${esc(RULE_COPY[r.rule] ?? "")}</div></div>`
        );
      })
      .join("") +
    "</div>";
}

async function ledger() {
  let events;
  try {
    events = await get("/api/ledger");
  } catch (err) {
    fail("#ledger", err);
    return;
  }
  const node = $("#ledger");
  if (!events.length) {
    node.innerHTML = '<p class="load">The ledger is empty.</p>';
    return;
  }
  node.innerHTML = events
    .map((e) => {
      const at = new Date(e.at).toLocaleTimeString("en-GB");
      const kind =
        e.event === "case.recovered"
          ? "rec"
          : e.event === "case.stopped" || e.event === "case.suppressed"
            ? "stop"
            : e.event === "case.action_deferred"
              ? "wait"
              : "";
      return (
        `<div class="e"><span class="t">${at}</span>` +
        `<span class="a">${esc(e.actor)}</span>` +
        `<span class="n ${kind}">${esc(e.event)} <span class="t">${esc(e.external_ref)}</span></span></div>`
      );
    })
    .join("");
}

/* ── go ───────────────────────────────────────────────────────────── */

watch();
nav();
story();
levers();
overview();
queue();
scheduled();
rules();
ledger();
tryPicker();


/* ── 03 · the interactive demonstration ─────────────────────────────────
   Renders whatever the ledger says. There is no branch here per failure
   reason and no canned outcome: the page asks the server to inject one
   payload, then reads back the decisions the real allocator made. If the
   taxonomy changes, this section changes with it and nobody edits this file. */

let tryBusy = false;

async function tryPicker() {
  const host = $("#try-picker");
  if (!host) return;
  let offers;
  try {
    offers = await get("/api/demo/causes");
  } catch (err) {
    fail("#try-picker", err);
    return;
  }

  host.innerHTML = "";
  offers.forEach((offer) => {
    const b = el("button", "pick");
    b.type = "button";
    b.setAttribute("aria-pressed", "false");
    b.dataset.reason = offer.error_reason;
    b.innerHTML =
      `<span class="pick-reason">${esc(offer.label)}</span>` +
      `<span class="pick-expect">${esc(offer.expect)}</span>`;
    b.addEventListener("click", () => runTry(offer, b));
    host.appendChild(b);
  });
}

async function runTry(offer, button) {
  if (tryBusy) return;
  tryBusy = true;

  document.querySelectorAll(".pick").forEach((b) => {
    b.setAttribute("aria-pressed", String(b === button));
    b.disabled = true;
  });

  const rail = $("#try-rail");
  const why = $("#try-why");
  rail.innerHTML = `<p class="load">Injecting a ${esc(offer.error_reason)} failure…</p>`;
  why.innerHTML = '<p class="load small muted">Deciding…</p>';

  try {
    const res = await fetch("/api/demo/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ error_reason: offer.error_reason }),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `simulate → ${res.status}`);
    }
    const { case_id: caseId } = await res.json();
    const data = await get(`/actions/case/${caseId}`);
    renderTry(data, offer);
  } catch (err) {
    fail("#try-rail", err);
    why.innerHTML = "";
  } finally {
    document.querySelectorAll(".pick").forEach((b) => { b.disabled = false; });
    tryBusy = false;
  }
}

function renderTry(data, offer) {
  const c = data.case;
  const events = data.ledger || [];
  const rail = $("#try-rail");
  rail.innerHTML = "";

  // Revealed one at a time so the sequence reads as a sequence. The delay is
  // presentation only -- every event below already happened, server-side,
  // before this function was called.
  events.forEach((event, i) => {
    const [what, detail] = narrate(event, c);
    const step = el("div", "step");
    step.style.animationDelay = `${i * 260}ms`;
    step.innerHTML =
      `<div class="when">${i + 1}</div>` +
      `<div class="what">${what}<span class="who">${esc(event.actor)}</span></div>` +
      (detail ? `<div class="small muted">${detail}</div>` : "");
    rail.appendChild(step);
  });

  const acted = (data.actions || [])[0];
  const refused = !acted;
  const verdict = refused
    ? "No action taken"
    : String(acted.type).replace(/_/g, " ");

  $("#try-why").innerHTML =
    `<div class="verdict${refused ? " refused" : ""}">` +
      `<div class="num serif">${esc(verdict)}</div>` +
      `<p class="small muted" style="margin:6px 0 0">${esc(offer.cause_label || c.cause || "")}</p>` +
    "</div>" +
    '<dl class="kv">' +
      row("cause", c.cause || "unclassified") +
      row("who can fix it", offer.who_can_fix) +
      row("retrying the same instrument", offer.retry_policy) +
      row("contacting the customer", offer.contact_ok ? "appropriate" : "not appropriate") +
      row("case status", c.status) +
    "</dl>" +
    `<p class="small muted" style="margin-top:14px">Simulated failure ` +
    `<span class="mono">${esc(c.external_ref)}</span>. Injected payload, real ` +
    `decisions, no Razorpay call.</p>`;
}

function row(k, v) {
  return `<dt>${esc(k)}</dt><dd>${esc(String(v ?? "—").replace(/_/g, " "))}</dd>`;
}
