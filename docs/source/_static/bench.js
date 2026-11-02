/* One comparison per problem. Names, the scale, and the number are separate columns. */
(function () {
  function compact(value) {
    var abs = Math.abs(value);
    if (abs >= 1000) {
      var scaled = Math.round((value / 1000) * 10) / 10;
      return String(scaled).replace(/\.0$/, "") + "k";
    }
    return String(Math.round(value));
  }

  function fraction(value, lo, hi) {
    if (!(hi > lo) || !(value > 0) || !(lo > 0)) {
      return 0;
    }
    return (Math.log(value) - Math.log(lo)) / (Math.log(hi) - Math.log(lo));
  }

  function draw(figure) {
    var mount = figure.querySelector(".bench-plot-mount");
    var dataEl = figure.querySelector(".bench-plot-data");
    if (!mount || !dataEl) {
      return;
    }
    var spec;
    try {
      spec = JSON.parse(dataEl.textContent);
    } catch (error) {
      return;
    }
    var points = (spec.points || []).filter(function (point) {
      return typeof point.value === "number" && isFinite(point.value) && point.value > 0;
    });
    if (points.length < 2) {
      return;
    }
    var values = points.map(function (point) { return point.value; });
    var lo = Math.min.apply(null, values);
    var hi = Math.max.apply(null, values);
    var list = document.createElement("ol");
    list.className = "bench-rank";
    points.forEach(function (point) {
      var item = document.createElement("li");
      if (point.best) {
        item.classList.add("best");
      }
      if (point.held) {
        item.classList.add("held");
      }
      if (point.failed) {
        item.classList.add("fail");
      }
      var name = document.createElement("a");
      name.href = "#" + point.id;
      name.textContent = point.entry;
      var track = document.createElement("span");
      track.className = "track";
      var dot = document.createElement("span");
      dot.className = "dot";
      var at = fraction(point.value, lo, hi) * 100;
      dot.style.left = at + "%";
      track.appendChild(dot);
      var num = document.createElement("span");
      num.className = "num";
      num.textContent = compact(point.value);
      item.appendChild(name);
      item.appendChild(track);
      item.appendChild(num);
      list.appendChild(item);
    });
    mount.replaceChildren(list);
  }

  function drawAll() {
    document.querySelectorAll(".bench-plot").forEach(draw);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", drawAll);
  } else {
    drawAll();
  }
}());

/* Column sort for the results tables. Empty cells stay last. */
(function () {
  function valueOf(cell) {
    return (cell ? cell.textContent : "").replace(/\s+/g, " ").trim();
  }

  function asNumber(text) {
    if (!text || text === "\u2014") {
      return null;
    }
    var cleaned = text.replace(/,/g, "").replace(/%$/, "");
    var value = Number(cleaned);
    return isFinite(value) ? value : null;
  }

  function sortTable(table, index, ascending) {
    var body = table.tBodies[0];
    if (!body) {
      return;
    }
    var rows = Array.prototype.slice.call(body.rows);
    rows.sort(function (left, right) {
      var a = valueOf(left.cells[index]);
      var b = valueOf(right.cells[index]);
      var an = asNumber(a);
      var bn = asNumber(b);
      if (an !== null && bn !== null) {
        return ascending ? an - bn : bn - an;
      }
      if (!a && b) {
        return 1;
      }
      if (a && !b) {
        return -1;
      }
      var order = a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
      return ascending ? order : -order;
    });
    rows.forEach(function (row) {
      body.appendChild(row);
    });
  }

  function bind(table) {
    var head = table.tHead;
    if (!head || !head.rows.length) {
      return;
    }
    Array.prototype.forEach.call(head.rows[0].cells, function (cell, index) {
      cell.addEventListener("click", function () {
        var ascending = cell.getAttribute("aria-sort") !== "ascending";
        Array.prototype.forEach.call(head.rows[0].cells, function (other) {
          other.removeAttribute("aria-sort");
        });
        cell.setAttribute("aria-sort", ascending ? "ascending" : "descending");
        sortTable(table, index, ascending);
      });
    });
  }

  function bindAll() {
    document.querySelectorAll("table.bench-sortable").forEach(bind);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindAll);
  } else {
    bindAll();
  }
}());
