from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_jlens_html(payload: dict[str, Any], output: str | Path) -> None:
    """Write a self-contained, offline J-Lens explorer."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    title = html.escape(str(payload.get("title", "Modern-MoE J-Lens")))
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--bg:#070a12;--panel:#101624;--line:#263249;--text:#e8eefc;--muted:#8290aa;
--cyan:#45e6d0;--violet:#9b7cff;--amber:#ffbd59}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 20% 0,#17203b 0,
var(--bg) 38%);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui}}
header{{position:sticky;top:0;z-index:5;padding:18px 24px;background:#070a12e8;
backdrop-filter:blur(16px);border-bottom:1px solid var(--line)}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:var(--muted)}} main{{padding:20px 24px}}
.cards{{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:12px;margin-bottom:18px}}
.card,.panel{{background:linear-gradient(145deg,#131a2aee,#0c111dee);border:1px solid var(--line);
border-radius:14px;box-shadow:0 12px 40px #0005}} .card{{padding:14px}}
.metric{{font-size:24px;font-weight:700;color:var(--cyan)}} .label{{color:var(--muted)}}
.toolbar{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:14px 0}}
button{{border:1px solid var(--line);background:#111a2c;color:var(--text);padding:8px 13px;
border-radius:9px;cursor:pointer}} button.active{{border-color:var(--cyan);color:var(--cyan)}}
.layout{{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:16px}}
.panel{{padding:14px;overflow:auto}} .grid{{display:grid;gap:3px;min-width:max-content}}
.corner,.token,.layer{{color:var(--muted);font-size:11px;padding:5px;overflow:hidden}}
.token{{writing-mode:vertical-rl;height:76px;text-align:left}} .layer{{text-align:right}}
.cell{{width:74px;height:38px;border:1px solid #ffffff12;border-radius:5px;padding:3px 5px;
overflow:hidden;white-space:nowrap;text-overflow:ellipsis;cursor:pointer;font-size:11px;
transition:.15s transform,.15s border-color}} .cell:hover,.cell.selected{{transform:scale(1.06);
border-color:var(--cyan);z-index:2}} .route{{font-size:9px;color:#c4b8ff}}
.detail h2{{font-size:16px;margin:0 0 8px}} .chips{{display:flex;flex-wrap:wrap;gap:6px}}
.chip{{padding:5px 8px;border-radius:999px;background:#172238;border:1px solid #293955}}
.rank{{color:var(--amber)}} .timeline{{margin-top:14px;border-top:1px solid var(--line);padding-top:12px}}
.bar{{display:grid;grid-template-columns:35px 1fr;gap:8px;align-items:center;margin:5px 0}}
.track{{height:8px;background:#1b2435;border-radius:9px;overflow:hidden}} .fill{{height:100%;
background:linear-gradient(90deg,var(--violet),var(--cyan))}}
.note{{margin-top:14px;color:var(--muted);font-size:12px}}
@media(max-width:900px){{.layout{{grid-template-columns:1fr}}.cards{{grid-template-columns:1fr 1fr}}}}
</style></head>
<body><header><h1>{title}</h1><div class="sub" id="prompt"></div></header>
<main><section class="cards" id="cards"></section>
<div class="toolbar"><span>读取方式</span><button data-mode="jlens" class="active">J-Lens</button>
<button data-mode="logit">Logit Lens</button><span class="sub">点击格子查看层轨迹与专家路由</span></div>
<div class="layout"><section class="panel"><div class="grid" id="grid"></div></section>
<aside class="panel detail" id="detail"></aside></div></main>
<script>
const D={serialized}; let mode="jlens", selected=[D.layers.length-1,D.tokens.length-1];
const esc=s=>String(s).replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",
'"':"&quot;","'":"&#39;"}}[c]));
document.querySelector("#prompt").textContent=D.prompt;
const avg=D.crystallization.filter(x=>x>=0).reduce((a,b)=>a+b,0)/
Math.max(1,D.crystallization.filter(x=>x>=0).length);
document.querySelector("#cards").innerHTML=[
["层数",D.layers.length],["Token",D.tokens.length],["Lens 样本",D.samples],
["平均结晶层",Number.isFinite(avg)?avg.toFixed(1):"—"]].map(x=>
`<div class=card><div class=label>${{x[0]}}</div><div class=metric>${{x[1]}}</div></div>`).join("");
function renderGrid(){{
 const g=document.querySelector("#grid"), cols=D.tokens.length;
 g.style.gridTemplateColumns=`48px repeat(${{cols}},74px)`;
 let out=`<div class=corner>层</div>`+D.tokens.map((t,i)=>
 `<div class=token title="${{esc(t)}}">${{i}} · ${{esc(t)}}</div>`).join("");
 D.layers.forEach((layer,li)=>{{
  out+=`<div class=layer>L${{layer}}</div>`;
  D[mode][li].forEach((cell,pi)=>{{
   const alpha=Math.max(.10,Math.min(.78,.16+cell.margin/12));
   const color=mode==="jlens"?`rgba(69,230,208,${{alpha}})`:`rgba(155,124,255,${{alpha}})`;
   const sel=li===selected[0]&&pi===selected[1]?" selected":"";
   out+=`<div class="cell${{sel}}" data-l="${{li}}" data-p="${{pi}}"
    style="background:${{color}}" title="${{esc(cell.top.join(" · "))}}">
    ${{esc(cell.top[0])}}<div class=route>E${{D.routes[li][pi].join(" · E")}}</div></div>`;
  }});
 }});
 g.innerHTML=out;
 g.querySelectorAll(".cell").forEach(el=>el.onclick=()=>{{
  selected=[+el.dataset.l,+el.dataset.p]; renderGrid(); renderDetail();
 }});
}}
function renderDetail(){{
 const [li,pi]=selected, layer=D.layers[li], cell=D[mode][li][pi];
 let trajectory=D.layers.map((l,i)=>({{l,word:D[mode][i][pi].top[0],
 margin:D[mode][i][pi].margin,routes:D.routes[i][pi]}}));
 document.querySelector("#detail").innerHTML=`<h2>位置 ${{pi}} · 层 ${{layer}}</h2>
 <div class=sub>输入 token：${{esc(D.tokens[pi])}}</div>
 <p class=label>Top concepts</p><div class=chips>${{cell.top.map((x,i)=>
 `<span class=chip><span class=rank>#${{i+1}}</span> ${{esc(x)}}</span>`).join("")}}</div>
 <p class=label>Top-2 路由专家</p><div class=chips>${{D.routes[li][pi].map(x=>
 `<span class=chip>E${{x}}</span>`).join("")}}</div><div class=timeline>
 <p class=label>跨层 top-1 轨迹</p>${{trajectory.map(x=>`<div class=bar><span>L${{x.l}}</span>
 <div><b>${{esc(x.word)}}</b><div class=track><div class=fill style="width:${{
 Math.min(100,10+x.margin*8)}}%"></div></div></div></div>`).join("")}}</div>
 <div class=note>“结晶层”表示最终预测 token 首次连续出现在 J-Lens top-k 的层；
它是读出指标，不等同于意识或完整思维。</div>`;
}}
document.querySelectorAll("button[data-mode]").forEach(b=>b.onclick=()=>{{
 mode=b.dataset.mode;document.querySelectorAll("button[data-mode]").forEach(x=>
 x.classList.toggle("active",x===b));renderGrid();renderDetail();
}});
renderGrid();renderDetail();
</script></body></html>"""
    output.write_text(document, encoding="utf-8")
