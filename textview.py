"""Build the interactive Current Text view for the GUI.

Kept separate from gui.py (which can only run inside Streamlit) so the
HTML-assembly logic is importable and unit-testable on its own. The returned
HTML is dropped into a Streamlit `components.html` iframe; words carry these
interactions, wired by the embedded JS:

  - hover           -> highlight the matching plot point + show next-token candidates
  - click / shift   -> cycle to next / previous sibling at a branch point
  - alt-click       -> split here and open a new sibling branch

Sibling/split actions reach the app by setting a `goto`/`splitat` query param
on the parent window (the same channel gui.py uses to persist node position).
"""

import html
import json

from coloring import (
    token_segments, hex_for_logprob, token_title, alt_bar_items, logprob_plot_svg,
)

_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><style>
 body{margin:0;font-family:'Source Sans Pro',system-ui,sans-serif;background:transparent;color:#9aa0a6;}
 .text{white-space:pre-wrap;line-height:1.75;font-size:1.02rem;}
 .text span{border-radius:2px;}
 .cur{font-weight:bold;}
 .branch{text-decoration:underline dotted;text-underline-offset:3px;text-decoration-thickness:2px;cursor:pointer;}
 .branch:hover{background:rgba(80,160,255,.28);}
 .hl{background:rgba(255,221,0,.45);}
 .altbar{min-height:104px;margin-top:10px;padding-top:8px;border-top:1px solid rgba(136,136,136,.25);}
 .alttitle{font-size:.7rem;color:#888;margin-bottom:5px;text-transform:uppercase;letter-spacing:.05em;}
 .altrow{display:flex;align-items:center;gap:7px;margin:1px 0;font-size:.8rem;}
 .altlabel{width:84px;flex:0 0 84px;font-family:ui-monospace,monospace;white-space:pre;overflow:hidden;text-overflow:ellipsis;text-align:right;color:#bbb;}
 .altrow.chosen .altlabel{color:#fff;font-weight:bold;}
 .altfill{height:10px;background:#6aa3ff;border-radius:2px;}
 .altrow.chosen .altfill{background:#ffd23c;}
 .altpct{color:#888;font-variant-numeric:tabular-nums;}
 .pt{cursor:pointer;transition:r .1s;}
 .plot{margin-top:10px;border-top:1px solid rgba(136,136,136,.25);padding-top:8px;}
 .hint{font-size:.7rem;color:#777;margin-top:8px;}
</style></head><body>
 <div class="text">__TEXT__</div>
 <div class="altbar" id="altbar"><div class="alttitle">hover a word for its next-token candidates</div></div>
 <div class="plot">__SVG__</div>
 <div class="hint">underlined = branch point · click cycles siblings, shift-click reverses, alt-click splits a new branch here</div>
 <script>
 var ALTS=__ALTS__;
 var APP=__APP__;
 function esc(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML;}
 function disp(s){return s.replace(/\\n/g,'\\u23ce').replace(/ /g,'\\u00b7')||'\\u2205';}
 function setPlot(i,on){var p=document.getElementById('p'+i);if(p){p.setAttribute('r',on?6:3);p.style.strokeWidth=on?2:0;}}
 function spanFor(i){return document.querySelector('.text span[data-i="'+i+'"]');}
 function showAlts(i){
   var a=ALTS[i],bar=document.getElementById('altbar');
   if(!a){return;}
   var max=a[0][1]||1;
   bar.innerHTML='<div class="alttitle">next-token candidates</div>'+a.map(function(r){
     var w=Math.max(2,Math.round(r[1]/max*150));
     return '<div class="altrow'+(r[2]?' chosen':'')+'"><span class="altlabel">'+esc(disp(r[0]))+'</span>'
       +'<span class="altfill" style="width:'+w+'px"></span>'
       +'<span class="altpct">'+(r[1]*100).toFixed(1)+'%</span></div>';
   }).join('');
 }
 function enter(i){var s=spanFor(i);if(s){s.classList.add('hl');}setPlot(i,true);showAlts(i);}
 function leave(i){var s=spanFor(i);if(s){s.classList.remove('hl');}setPlot(i,false);}
 function nav(param,value){
   // The component runs in a sandboxed srcdoc iframe, so window.parent.location
   // is unreadable (opaque origin). Rebuild the app URL from document.referrer
   // (the parent page, which IS readable) plus the current query params handed
   // in from Python, then navigate the top frame.
   var u;
   try{u=new URL(document.referrer);}catch(e){return;}
   Object.keys(APP).forEach(function(k){u.searchParams.set(k,APP[k]);});
   u.searchParams.delete('goto');u.searchParams.delete('splitat');
   u.searchParams.set(param,value);
   window.top.location.href=u.toString();
 }
 document.querySelectorAll('.text span').forEach(function(s){
   var i=s.dataset.i;
   s.addEventListener('mouseenter',function(){enter(i);});
   s.addEventListener('mouseleave',function(){leave(i);});
   s.addEventListener('click',function(e){
     if(e.altKey && s.dataset.split!==undefined){
       e.preventDefault();
       var off=parseInt(s.dataset.split,10);
       if(off>0){nav('splitat',off);}
       return;
     }
     if(s.classList.contains('branch')){
       e.preventDefault();
       var sibs=(s.dataset.sibs||'').split(',').filter(Boolean);
       if(sibs.length<2){return;}
       var pos=parseInt(s.dataset.pos||'0',10);
       var next=(pos+(e.shiftKey?-1:1)+sibs.length)%sibs.length;
       nav('goto',sibs[next]);
     }
   });
 });
 document.querySelectorAll('.pt').forEach(function(p){
   var i=p.dataset.idx;
   p.addEventListener('mouseenter',function(){enter(i);});
   p.addEventListener('mouseleave',function(){leave(i);});
 });
 </script></body></html>"""


def build_text_component(loom, app_params=None):
    """Return (html, has_logprobs) for the path root..current_node.

    `app_params` is the current Streamlit query-param dict (file/node); it is
    embedded so the in-iframe JS can rebuild the parent URL when a word click
    navigates (the sandbox blocks reading the parent location directly).

    Colors every token by surprisal, marks branch points (a node's first token
    when its parent has more than one child), attaches per-token next-token
    candidates for the hover bar, and links current-node tokens to the plot via
    a single global token index.
    """
    path = loom.tree.get_path_to_node(loom.current_node.id)
    spans = []
    alts = {}            # global token index -> [[token, prob, is_chosen], ...]
    plot_points = []     # (global index, logprob) for current-node tokens
    gidx = 0
    any_lp = False

    for node in path:
        is_current = node is loom.current_node
        segs = token_segments(node.text, node.logprobs)
        top = (node.logprobs or {}).get("top_logprobs")
        aligned_top = top if (top and len(top) == len(segs)) else None

        sibs = None
        pos = 0
        sib_nodes = loom.siblings(node.id)
        if len(sib_nodes) > 1:
            sibs = [c.id for c in sib_nodes]
            pos = next((j for j, c in enumerate(sib_nodes) if c.id == node.id), 0)

        char_off = 0
        for k, (tok, lp) in enumerate(segs):
            classes = []
            styles = ""
            title = ""
            attrs = f'data-i="{gidx}"'

            if lp is not None:
                any_lp = True
                styles = f"color:{hex_for_logprob(lp)}"
                title = token_title(lp)
                if is_current:
                    plot_points.append((gidx, lp))

            if is_current:
                classes.append("cur")
                attrs += f' data-split="{char_off}"'

            if aligned_top and aligned_top[k]:
                items = alt_bar_items(aligned_top[k])
                if items:
                    alts[gidx] = [[t, p, (t == tok)] for t, p in items]

            if sibs is not None and k == 0:
                classes.append("branch")
                attrs += f' data-sibs="{",".join(sibs)}" data-pos="{pos}"'
                title = f"branch point · {len(sibs)} options · click cycles siblings, shift-click reverses"

            cls = f' class="{" ".join(classes)}"' if classes else ""
            sty = f' style="{styles}"' if styles else ""
            ttl = f' title="{html.escape(title)}"' if title else ""
            spans.append(f"<span{cls}{sty}{ttl} {attrs}>{html.escape(tok)}</span>")
            char_off += len(tok)
            gidx += 1

    component = (_TEMPLATE
                 .replace("__TEXT__", "".join(spans))
                 .replace("__SVG__", logprob_plot_svg(plot_points))
                 .replace("__ALTS__", json.dumps(alts).replace("</", "<\\/"))
                 .replace("__APP__", json.dumps(app_params or {})))
    return component, any_lp
