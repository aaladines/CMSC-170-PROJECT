document.addEventListener("DOMContentLoaded", () => {
  if (typeof MODEL_RESULTS === "undefined") {
    console.error("MODEL_RESULTS not found. Check that tables_and_charts/results.js is loaded.");
    return;
  }

  renderYearlyTable(MODEL_RESULTS);
  renderAveragesTable(MODEL_RESULTS);
});

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