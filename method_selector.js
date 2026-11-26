function connectMethodChart(plotId) {
  const state = JSON.parse(document.getElementById("method-state").textContent);
  const plot = document.getElementById(plotId);

  function announce() {
    if (window.parent !== window) {
      window.parent.postMessage({
        type: "method-chart-ready",
        records: state.records.map((item) => item.record),
      }, "*");
    }
  }

  window.addEventListener("message", (event) => {
    if (event.source !== window.parent) return;
    if (event.data?.type === "method-chart-request") return announce();
    if (event.data?.type !== "method-record-selected") return;

    const index = Number(event.data.index);
    if (!Number.isInteger(index) || index < 0 || index >= state.records.length) return;
    const selected = state.records[index];

    Plotly.react(plot, [...state.base_data, ...selected.traces],
      {...state.layout, ...selected.layout}, {responsive: true}).catch((error) => console.error(error));
  });

  announce();
}

document.addEventListener("DOMContentLoaded", () => {
  const selector = document.getElementById("method-record");
  if (!selector) return;

  const frames = [...document.querySelectorAll(".method-chart iframe")];
  const summary = document.getElementById("method-record-summary");
  let records = [];

  function send(frame) {
    frame.contentWindow.postMessage({type: "method-record-selected", index: Number(selector.value)}, "*");
  }

  function describe() {
    const record = records[Number(selector.value)];
    summary.textContent = `${record.crop} · ${record.region} · Q${record.quarter} · Actual: ${record.actual === 1 ? "Increase (1)" : "Decrease / unchanged (0)"}`;
  }

  window.addEventListener("message", (event) => {
    const frame = frames.find((item) => item.contentWindow === event.source);
    if (!frame || event.data?.type !== "method-chart-ready") return;

    if (!records.length) {
      const incoming = event.data.records;
      if (!Array.isArray(incoming) || !incoming.length || !incoming.every((r) =>
        typeof r.crop === "string" && typeof r.region === "string" &&
        Number.isInteger(r.quarter) && r.quarter >= 1 && r.quarter <= 4 &&
        (r.actual === 0 || r.actual === 1))) return;

      records = incoming;
      selector.replaceChildren();
      const crops = [...new Set(records.map((record) => record.crop))].sort((a,b) => a.localeCompare(b));

      crops.forEach((crop) => {
        const group = document.createElement("optgroup");
        group.label = crop;

        records.forEach((record,index) => {
          if (record.crop === crop) group.append(new Option(`${record.crop} · ${record.region} · Q${record.quarter}`,index));
        });

        selector.append(group);
      });

      selector.value = "0";
      selector.disabled = false;
      describe();
    }

    send(frame);
  });

  selector.addEventListener("change", () => {
    describe();
    frames.forEach(send);
  });

  frames.forEach((frame) => {
    const request = () => frame.contentWindow.postMessage({type: "method-chart-request"}, "*");
    frame.addEventListener("load", request);
    request();
  });
});
