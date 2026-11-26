document.addEventListener("DOMContentLoaded", () => {
  if (typeof MODEL_RESULTS === "undefined") {
    console.error("MODEL_RESULTS not found. Check that tables_and_charts/results.js is loaded.");
    return;
  }

  renderYearlyTable(MODEL_RESULTS);
  renderAveragesTable(MODEL_RESULTS);
  renderConfusionMatrices(CONFUSION_MATRICES);
});

function renderConfusionMatrices(matrices) {
  const container = document.getElementById("confusion-matrices");
  const classes = ["Decrease / stagnant (0)", "Increase (1)"];

  Object.entries(matrices).forEach(([name, matrix]) => {
    const card = document.createElement("div");
    card.className = "table-wrap";
    const table = document.createElement("table");
    table.className = "confusion-matrix";
    table.createCaption().textContent = name;

    const header = table.createTHead().insertRow();
    ["Actual ↓ / Predicted →", ...classes].forEach((label) => {
      const th = document.createElement("th");
      th.scope = "col";
      th.textContent = label;
      header.appendChild(th);
    });

    const body = table.createTBody();
    matrix.forEach((counts, actual) => {
      const row = body.insertRow();
      const label = document.createElement("th");
      label.scope = "row";
      label.textContent = classes[actual];
      row.appendChild(label);
      const total = counts.reduce((sum, count) => sum + count, 0);

      counts.forEach((count, predicted) => {
        const percentage = total === 0 ? 0 : (count / total) * 100;
        const cell = row.insertCell();
        cell.style.backgroundColor = actual === predicted
          ? `hsl(145, 35%, ${97 - percentage * 0.3}%)`
          : `hsl(43, 90%, ${97 - percentage * 0.25}%)`;
        const value = document.createElement("strong");
        value.textContent = count;
        const percent = document.createElement("span");
        percent.textContent = `${percentage.toFixed(1)}%`;
        cell.append(value, percent);
      });
    });

    card.appendChild(table);
    container.appendChild(card);
  });
}

function renderYearlyTable(data) {
  const table = document.getElementById("year-results-table");
  const metrics = ["accuracy", "precision", "recall", "f1_score"];

  data.forEach((entry) => {
    const tbody = document.createElement("tbody");

    const maxPerMetric = {};
    metrics.forEach((m) => {
      maxPerMetric[m] = Math.max(
        ...entry.models.map((model) => Number(model[m].toFixed(4)))
      );
    });

    entry.models.forEach((model, idx) => {
      const tr = document.createElement("tr");

      if (idx === 0) {
        const th = document.createElement("th");
        th.scope = "rowgroup";
        th.rowSpan = entry.models.length;
        th.className = "yr";
        th.textContent = entry.year;
        tr.appendChild(th);
      }

      const tdModel = document.createElement("td");
      tdModel.className = "model";
      tdModel.textContent = model.name;
      tr.appendChild(tdModel);

      const tdCount = document.createElement("td");
      tdCount.textContent = entry.n_data;
      tr.appendChild(tdCount);

      metrics.forEach((m) => {
        const td = document.createElement("td");
        const formatted = model[m].toFixed(4);
        td.textContent = formatted;

        if (Number(formatted) === maxPerMetric[m]) {
          td.className = "best";
        }
        tr.appendChild(td);
      });

      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
  });
}

function renderAveragesTable(data) {
  const tbody = document.getElementById("avg-table-body");
  const metrics = ["accuracy", "precision", "recall", "f1_score"];
  const stats = {};

  data.forEach((entry) => {
    entry.models.forEach((model) => {
      if (!stats[model.name]) {
        stats[model.name] = {
          config: model.config,
          count: 0,
          accuracy: 0,
          precision: 0,
          recall: 0,
          f1_score: 0,
        };
      }
      stats[model.name].count += 1;
      metrics.forEach((m) => {
        stats[model.name][m] += model[m];
      });
    });
  });

  Object.entries(stats).forEach(([name, info]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${name}</td>
      <td>${info.config}</td>
      <td>${(info.accuracy / info.count).toFixed(4)}</td>
      <td>${(info.precision / info.count).toFixed(4)}</td>
      <td>${(info.recall / info.count).toFixed(4)}</td>
      <td>${(info.f1_score / info.count).toFixed(4)}</td>
    `;
    tbody.appendChild(tr);
  });
}