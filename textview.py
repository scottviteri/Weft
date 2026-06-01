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
    surprisal_bits, gen_params_label,
)

_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><style>
 body{margin:0;font-family:'Source Sans Pro',system-ui,sans-serif;background:transparent;color:#9aa0a6;}
 .text{white-space:pre-wrap;line-height:1.75;font-size:1.02rem;}
 .text span{border-radius:2px;}
 .cur{font-weight:bold;}
 /* Visible 8px pill, but content-box padding makes the click target ~18px. */
 .fork{display:inline-block;box-sizing:content-box;width:8px;height:1em;
   padding:2px 5px;margin:0 1px;vertical-align:-0.2em;cursor:pointer;
   background:#6aa3ff;background-clip:content-box;border-radius:3px;
   box-shadow:0 0 4px rgba(106,163,255,.7);transition:background .12s,box-shadow .12s;}
 .fork:hover{background:#ffd23c;background-clip:content-box;box-shadow:0 0 7px rgba(255,210,60,.95);}
 .fork:active{background:#ffb300;background-clip:content-box;}
 .hl{background:rgba(255,221,0,.45);}
 .altbar{min-height:104px;margin-top:10px;padding-top:8px;border-top:1px solid rgba(136,136,136,.25);}
 .alttitle{font-size:.7rem;color:#888;margin-bottom:5px;text-transform:uppercase;letter-spacing:.05em;}
 .altrow{display:flex;align-items:center;gap:7px;margin:1px 0;font-size:.8rem;}
 .altlabel{width:84px;flex:0 0 84px;font-family:ui-monospace,monospace;white-space:pre;overflow:hidden;text-overflow:ellipsis;text-align:right;color:#bbb;}
 .altrow.chosen .altlabel{color:#fff;font-weight:bold;}
 .altfill{height:10px;background:#6aa3ff;border-radius:2px;}
 .altrow.chosen .altfill{background:#ffd23c;}
 .altpct{color:#888;font-variant-numeric:tabular-nums;}
 .surp{margin-top:9px;font-size:.8rem;color:#aab;font-variant-numeric:tabular-nums;}
 .surp b{color:#ffd23c;font-weight:600;}
 .pt{cursor:pointer;transition:r .1s;}
 .plot{margin-top:10px;border-top:1px solid rgba(136,136,136,.25);padding-top:8px;}
 .hint{font-size:.7rem;color:#777;margin-top:8px;}
</style></head><body>
 <div class="text">__TEXT__</div>
 <div class="altbar" id="altbar"><div class="alttitle">hover a word for its next-token candidates</div></div>
 <div class="surp" id="surp">__STATS__</div>
 <div class="plot">__SVG__</div>
 <div class="hint">blue bar = branch point (between the shared and diverging token) · click cycles siblings, shift-click reverses · alt-click a word to split a new branch there</div>
 <div class="hint" style="color:#6aa3ff;font-family:ui-monospace,monospace;">iframe node: __NODEID__</div>
 <script>
 var ALTS=__ALTS__;
 var CUM=__CUM__;
 var STATS=__STATSJS__;
 var CNT=0;
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
 function showSurp(i){
   var c=CUM[i],el=document.getElementById('surp');
   if(!el){return;}
   if(c){el.innerHTML='this token <b>'+c[0].toFixed(1)+'</b> bits · cumulative to here <b>'
     +c[1].toFixed(1)+'</b> bits'+(c[0]>0?' (1 in '+Math.round(Math.pow(2,c[0])).toLocaleString()+')':'');}
 }
 function restoreSurp(){var el=document.getElementById('surp');if(el){el.innerHTML=STATS;}}
 function enter(i){var s=spanFor(i);if(s){s.classList.add('hl');}setPlot(i,true);showAlts(i);showSurp(i);}
 function leave(i){var s=spanFor(i);if(s){s.classList.remove('hl');}setPlot(i,false);restoreSurp();}
 // In-text clicks reach the app by writing into a hidden Streamlit text_input in
 // the parent (allow-same-origin lets us reach it) and committing it, which
 // reruns over the existing WebSocket — no page reload. A counter keeps each
 // command unique so repeats still register. There is deliberately no reload
 // fallback: if the input can't be reached the click is simply a no-op.
 // [weft] console logging is left in to make this debuggable from DevTools.
 function sendCmd(kind,value){
   console.log('[weft] sendCmd', kind, value);
   try{
     var pw=window.parent;
     if(!pw||!pw.document){console.warn('[weft] parent document not accessible');return;}
     var pdoc=pw.document;
     // Streamlit tags keyed widgets with an `st-key-<key>` class on their
     // container; that's a more reliable handle than the input's aria-label.
     var input=pdoc.querySelector('.st-key-weft_cmd input')||pdoc.querySelector('input[aria-label="weft_cmd"]');
     if(!input){
       console.warn('[weft] weft_cmd input NOT found. Inputs in parent:');
       pdoc.querySelectorAll('input').forEach(function(el,i){
         var box=el.closest('[class*="st-key"]');
         console.log('  ['+i+'] aria='+el.getAttribute('aria-label')+
           ' type='+el.type+' container='+(box?box.className:'(none)'));
       });
       return;
     }
     CNT++;
     var val=kind+':'+value+':'+CNT;
     var setter=Object.getOwnPropertyDescriptor(pw.HTMLInputElement.prototype,'value').set;
     setter.call(input,val);
     // The native setter defeats React's value tracker, so dispatch input to
     // make React notice the change, then Enter to commit (Streamlit reruns).
     input.dispatchEvent(new pw.Event('input',{bubbles:true}));
     ['keydown','keypress','keyup'].forEach(function(t){
       input.dispatchEvent(new pw.KeyboardEvent(t,{key:'Enter',code:'Enter',keyCode:13,which:13,bubbles:true}));
     });
     console.log('[weft] committed value=', val, '| input.value now=', input.value);
   }catch(e){console.error('[weft] sendCmd error', e);}
 }
 // Tokens: hover to highlight + show candidates; alt-click to split here.
 document.querySelectorAll('.text span[data-i]').forEach(function(s){
   var i=s.dataset.i;
   s.addEventListener('mouseenter',function(){enter(i);});
   s.addEventListener('mouseleave',function(){leave(i);});
   s.addEventListener('click',function(e){
     if(e.altKey && s.dataset.split!==undefined){
       e.preventDefault();
       var off=parseInt(s.dataset.split,10);
       if(off>0){sendCmd('split',off);}
     }
   });
 });
 // Fork markers (between the shared and diverging token): click to cycle siblings.
 console.log('[weft] fork markers found:', document.querySelectorAll('.fork').length);
 document.querySelectorAll('.fork').forEach(function(f){
   f.addEventListener('click',function(e){
     e.preventDefault();
     var sibs=(f.dataset.sibs||'').split(',').filter(Boolean);
     console.log('[weft] fork click; sibs=', sibs, 'pos=', f.dataset.pos, 'shift=', e.shiftKey);
     if(sibs.length<2){return;}
     var pos=parseInt(f.dataset.pos||'0',10);
     var next=(pos+(e.shiftKey?-1:1)+sibs.length)%sibs.length;
     sendCmd('goto',sibs[next]);
   });
 });
 document.querySelectorAll('.pt').forEach(function(p){
   var i=p.dataset.idx;
   p.addEventListener('mouseenter',function(){enter(i);});
   p.addEventListener('mouseleave',function(){leave(i);});
 });
 </script></body></html>"""


def build_text_component(loom):
    """Return (html, has_logprobs) for the path root..current_node.

    Colors every token by surprisal, inserts a clickable fork marker between the
    shared token and the diverging token at each branch point, attaches per-token
    next-token candidates for the hover bar, and links current-node tokens to the
    plot via a single global token index.
    """
    path = loom.tree.get_path_to_node(loom.current_node.id)
    spans = []
    alts = {}            # global token index -> [[token, prob, is_chosen], ...]
    cum = {}             # global token index -> [token_bits, cumulative_bits_from_root]
    plot_points = []     # (global index, logprob) for the plot window
    gidx = 0
    any_lp = False
    cum_bits = 0.0       # running total surprisal (bits) from root
    n_scored = 0         # tokens that carry a logprob

    # The plot covers from just before the most recent branch point (the deepest
    # fork on the path) through the current node, so you can see the logprob
    # trajectory leading into and out of the fork rather than only this node.
    branch_js = [j for j, n in enumerate(path) if len(loom.siblings(n.id)) > 1]
    if branch_js:
        plot_start_j = max(0, max(branch_js) - 1)
        branch_node = path[max(branch_js)]
    else:
        plot_start_j = len(path) - 1
        branch_node = None
    mark_idx = None

    for j, node in enumerate(path):
        is_current = node is loom.current_node
        in_plot = j >= plot_start_j
        if branch_node is not None and node is branch_node:
            mark_idx = gidx  # global index of the fork's first token
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
                t_bits = surprisal_bits(lp)
                cum_bits += t_bits
                n_scored += 1
                cum[gidx] = [round(t_bits, 3), round(cum_bits, 3)]
                if in_plot:
                    plot_points.append((gidx, lp))

            if is_current:
                classes.append("cur")
                attrs += f' data-split="{char_off}"'

            if aligned_top and aligned_top[k]:
                items = alt_bar_items(aligned_top[k])
                if items:
                    alts[gidx] = [[t, p, (t == tok)] for t, p in items]

            # A fork marker sits BETWEEN the shared (parent) token and this
            # node's first (diverging) token, rather than recoloring the token.
            if sibs is not None and k == 0:
                ftitle = (f"branch point · {len(sibs)} options · "
                          f"click cycles siblings, shift-click reverses")
                spans.append(
                    f'<span class="fork" data-sibs="{",".join(sibs)}" '
                    f'data-pos="{pos}" title="{html.escape(ftitle)}"></span>'
                )

            cls = f' class="{" ".join(classes)}"' if classes else ""
            sty = f' style="{styles}"' if styles else ""
            ttl = f' title="{html.escape(title)}"' if title else ""
            spans.append(f"<span{cls}{sty}{ttl} {attrs}>{html.escape(tok)}</span>")
            char_off += len(tok)
            gidx += 1

    # Default stats line: total surprisal of the whole branch (root -> current)
    # plus the average bits/token, and the temperature the current node was
    # produced at (surprisal only compares fairly across equal temperatures).
    if n_scored:
        per_tok = cum_bits / n_scored
        stats = (f"branch surprisal <b>{cum_bits:.1f}</b> bits over {n_scored} tokens "
                 f"· <b>{per_tok:.2f}</b> bits/token")
        plabel = gen_params_label(loom.current_node.logprobs)
        if plabel:
            stats += f" · this node: {plabel}"
    else:
        stats = "no logprobs yet — score this text to measure its surprisal"

    component = (_TEMPLATE
                 .replace("__TEXT__", "".join(spans))
                 .replace("__SVG__", logprob_plot_svg(plot_points, mark_idx=mark_idx))
                 .replace("__NODEID__", html.escape(loom.current_node.id))
                 .replace("__STATS__", stats)
                 .replace("__STATSJS__", json.dumps(stats))
                 .replace("__CUM__", json.dumps(cum).replace("</", "<\\/"))
                 .replace("__ALTS__", json.dumps(alts).replace("</", "<\\/")))
    return component, any_lp
