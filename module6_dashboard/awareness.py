"""
Awareness, Resources and Donate pages (pink-ribbon theme).
Illustrations are inline SVG (data-URI images), so everything works offline.
Medical facts follow the WHO breast-cancer and cancer fact sheets.
"""
import base64
import glob
import html as _html
import os

import streamlit as st

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _md(s):
    """Render an HTML snippet. Blank lines / indentation would make Markdown
    break the HTML block, so they are stripped first."""
    st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()),
                unsafe_allow_html=True)


def _img(svg, style=""):
    b64 = base64.b64encode(svg.encode('utf-8')).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" style="{style}" alt="" />'


def ribbon_svg(small=False):
    sw = 15
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 140">'
        '<defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="120" y2="140">'
        '<stop offset="0" stop-color="#FFB3D1"/><stop offset="1" stop-color="#E5195F"/></linearGradient></defs>'
        f'<g fill="none" stroke="url(#g)" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M60 78 C18 46 30 10 60 10 C90 10 102 46 60 78"/>'
        '<path d="M60 78 L34 126"/><path d="M60 78 L86 126"/></g></svg>')


def solidarity_svg():
    skins = ['#F6D5C0', '#E8B48F', '#C68642', '#8D5524', '#F1C27D']
    hairs = ['#3B2A20', '#1B1B1B', '#5A3825', '#101010', '#8A5A2B']
    dress = ['#FF8FB8', '#C77DFF', '#FF5C9A', '#7AA2FF', '#FFB86B']
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 190">',
             '<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
             '<stop offset="0" stop-color="#3A1C4A"/><stop offset="1" stop-color="#1B2A5A"/></linearGradient></defs>',
             '<rect width="640" height="190" rx="22" fill="url(#bg)"/>',
             '<circle cx="60" cy="40" r="70" fill="#FF5C9A" opacity=".10"/>',
             '<circle cx="590" cy="150" r="90" fill="#C77DFF" opacity=".12"/>']
    for i in range(5):
        cx = 90 + i * 115
        parts.append(
            f'<ellipse cx="{cx}" cy="74" rx="29" ry="33" fill="{hairs[i]}"/>'
            f'<path d="M{cx-50} 190 Q{cx-50} 116 {cx} 112 Q{cx+50} 116 {cx+50} 190Z" fill="{dress[i]}"/>'
            f'<rect x="{cx-7}" y="96" width="14" height="20" fill="{skins[i]}"/>'
            f'<circle cx="{cx}" cy="80" r="22" fill="{skins[i]}"/>'
            f'<path d="M{cx-23} 74 Q{cx} 46 {cx+23} 74 Q{cx} 62 {cx-23} 74Z" fill="{hairs[i]}"/>'
            f'<circle cx="{cx-8}" cy="82" r="2" fill="#2b2b2b"/><circle cx="{cx+8}" cy="82" r="2" fill="#2b2b2b"/>'
            f'<path d="M{cx-7} 91 Q{cx} 97 {cx+7} 91" stroke="#2b2b2b" stroke-width="2" fill="none" stroke-linecap="round"/>'
            f'<g transform="translate({cx-9} 128) scale(.15)" fill="none" stroke="#fff" stroke-width="15" '
            f'stroke-linecap="round"><path d="M60 78 C18 46 30 10 60 10 C90 10 102 46 60 78"/>'
            f'<path d="M60 78 L34 126"/><path d="M60 78 L86 126"/></g>')
    parts.append('</svg>')
    return ''.join(parts)


def wave_svg(color="#080C14"):
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 90" preserveAspectRatio="none">'
            f'<path fill="{color}" d="M0 40 C240 90 480 0 720 40 C960 80 1200 10 1440 50 L1440 90 L0 90Z"/></svg>')


AW_CSS = """
<style>
.aw-hero{position:relative;overflow:hidden;text-align:center;padding:3.2rem 1.5rem 5.5rem;
 background:radial-gradient(circle at 20% 20%,#FF5C9A55,transparent 45%),
 radial-gradient(circle at 85% 30%,#C77DFF44,transparent 45%),
 linear-gradient(135deg,#2B0F3A 0%,#4A1550 40%,#13204A 100%);}
.aw-hero h1{font-family:'Playfair Display',serif;font-size:3rem;color:#fff;line-height:1.15;margin:.6rem 0}
.aw-hero h1 em{background:linear-gradient(90deg,#FF8FB8,#FFD1E3);-webkit-background-clip:text;
 -webkit-text-fill-color:transparent;font-style:italic}
.aw-sub{color:#F3D9E8;max-width:720px;margin:0 auto 1.4rem;font-size:1.08rem;line-height:1.7}
.aw-tag{display:inline-block;padding:.35rem 1rem;border-radius:99px;background:#ffffff18;
 border:1px solid #ffffff33;color:#FFD1E3;font-size:.8rem;letter-spacing:.12em;text-transform:uppercase}
.aw-float{position:absolute;opacity:.55;animation:awfloat 7s ease-in-out infinite;user-select:none}
@keyframes awfloat{0%,100%{transform:translateY(0) rotate(0)}50%{transform:translateY(-22px) rotate(12deg)}}
@keyframes awpulse{0%,100%{transform:scale(1)}50%{transform:scale(1.07)}}
.aw-ribbon{animation:awpulse 3.2s ease-in-out infinite;filter:drop-shadow(0 8px 24px #FF5C9A88)}
.aw-btn{display:inline-block;margin:.3rem;padding:.75rem 1.6rem;border-radius:99px;font-weight:600;
 text-decoration:none!important;transition:transform .2s,box-shadow .2s}
.aw-btn:hover{transform:translateY(-3px);box-shadow:0 10px 28px #FF5C9A66}
.aw-btn.p{background:linear-gradient(90deg,#FF5C9A,#C77DFF);color:#fff!important}
.aw-btn.s{border:1.5px solid #FF8FB8;color:#FFD1E3!important}
.aw-wave{display:block;width:100%;height:70px;margin-top:-70px;position:relative}
.aw-chips{display:flex;flex-wrap:wrap;gap:.7rem;justify-content:center;margin-top:1.2rem}
.aw-chip{background:#ffffff14;border:1px solid #ffffff2b;border-radius:14px;padding:.7rem 1.1rem;color:#fff;min-width:150px}
.aw-chip b{display:block;font-size:1.5rem;color:#FF8FB8;font-family:'Playfair Display',serif}
.aw-chip span{font-size:.78rem;color:#E9C9DA}
.aw-sec{max-width:1150px;margin:0 auto;padding:2.2rem 1.5rem 0}
.aw-h{font-family:'Playfair Display',serif;font-size:2rem;color:#fff;text-align:center;margin-bottom:.3rem}
.aw-h:after{content:'';display:block;width:70px;height:4px;margin:.6rem auto 0;border-radius:4px;
 background:linear-gradient(90deg,#FF5C9A,#C77DFF)}
.aw-p{color:#9FB0C8;text-align:center;max-width:760px;margin:.6rem auto 1.6rem;line-height:1.7}
.aw-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1.1rem}
.aw-card{position:relative;background:linear-gradient(160deg,#141B2B,#0F1623);border:1px solid #ffffff12;
 border-radius:18px;padding:1.4rem;transition:transform .25s,border-color .25s,box-shadow .25s;overflow:hidden}
.aw-card:hover{transform:translateY(-6px);border-color:#FF5C9A88;box-shadow:0 14px 34px #FF5C9A22}
.aw-card:before{content:'';position:absolute;top:-40px;right:-40px;width:110px;height:110px;border-radius:50%;
 background:radial-gradient(circle,#FF5C9A33,transparent 70%)}
.aw-ico{width:58px;height:58px;border-radius:16px;display:flex;align-items:center;justify-content:center;
 font-size:1.9rem;margin-bottom:.8rem;background:linear-gradient(135deg,#FF5C9A55,#C77DFF44)}
.aw-card h4{color:#fff;font-size:1.08rem;margin-bottom:.4rem}
.aw-card p{color:#A9B7CC;font-size:.9rem;line-height:1.65}
.aw-card.g .aw-ico{background:linear-gradient(135deg,#2DD4A055,#3B82F644)}
.aw-card.g:hover{border-color:#2DD4A088;box-shadow:0 14px 34px #2DD4A022}
.aw-card.g:before{background:radial-gradient(circle,#2DD4A033,transparent 70%)}
.aw-num{position:absolute;top:.9rem;right:1.1rem;font-family:'Playfair Display',serif;font-size:2.2rem;color:#ffffff10}
.aw-hope{margin:2.2rem auto 0;max-width:1000px;text-align:center;padding:1.8rem;border-radius:22px;
 background:linear-gradient(120deg,#FF5C9A33,#C77DFF22,#2DD4A022);border:1px solid #ffffff1f}
.aw-hope q{display:block;color:#fff;font-family:'Playfair Display',serif;font-size:1.45rem;line-height:1.5}
.aw-hope small{color:#C9D3E0}
.aw-note{max-width:900px;margin:1.4rem auto 0;padding:.9rem 1.2rem;border-radius:12px;background:#FFB86B14;
 border:1px solid #FFB86B44;color:#FFD9A8;font-size:.85rem;line-height:1.6;text-align:center}
.aw-gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem}
.aw-link{display:inline-block;margin-top:.8rem;padding:.5rem 1.1rem;border-radius:99px;font-size:.85rem;font-weight:600;
 background:#FF5C9A22;border:1px solid #FF5C9A77;color:#FFB3D1!important;text-decoration:none!important}
.aw-link:hover{background:#FF5C9A55}
.aw-tier{border-left:4px solid #FF5C9A}
.aw-pledge{display:inline-block;margin:.3rem;padding:.45rem .9rem;border-radius:99px;background:#FF5C9A1f;
 border:1px solid #FF5C9A55;color:#FFD1E3;font-size:.85rem}
</style>
"""

FLOATERS = [('🌸', '6%', '14%', '0s', '2rem'), ('🎀', '88%', '12%', '1.4s', '2.4rem'),
            ('💗', '15%', '62%', '2.2s', '1.7rem'), ('🌷', '80%', '58%', '.8s', '2rem'),
            ('✨', '48%', '8%', '3s', '1.6rem'), ('🦋', '93%', '38%', '2.6s', '1.8rem'),
            ('🌺', '3%', '40%', '3.4s', '1.8rem')]


def _floaters():
    return ''.join(
        f'<span class="aw-float" style="left:{x};top:{y};animation-delay:{d};font-size:{s}">{e}</span>'
        for e, x, y, d, s in FLOATERS)


def _cards(items, cls=""):
    out = ['<div class="aw-grid">']
    for i, (icon, title, text) in enumerate(items, 1):
        out.append(f'<div class="aw-card {cls}"><div class="aw-num">{i:02d}</div>'
                   f'<div class="aw-ico">{icon}</div><h4>{title}</h4><p>{text}</p></div>')
    out.append('</div>')
    return ''.join(out)


def _wave(color):
    return _img(wave_svg(color), "display:block;width:100%;height:70px;margin-top:-69px;position:relative")


# --------------------------------------------------------------------------
# HOME (awareness)
# --------------------------------------------------------------------------
CAUSES = [
    ('⏳', 'Age', 'Risk rises as we get older. Most breast cancers are found in women over 40, '
                  'which is why regular check-ups matter more with age.'),
    ('🧬', 'Inherited genes', 'Changes in genes such as BRCA1, BRCA2 and PALB2 greatly raise risk. '
                              'They are uncommon, but worth knowing about if cancer runs in your family.'),
    ('👪', 'Family history', 'A close relative with breast or ovarian cancer raises your risk - '
                             'yet most women diagnosed have no known family history.'),
    ('🌙', 'Hormones & reproductive history', 'Periods starting before 12, menopause after 55, first pregnancy '
                                              'after 30 or no children mean longer oestrogen exposure. '
                                              'Long-term hormone therapy can also add risk.'),
    ('🍷', 'Alcohol & tobacco', 'Harmful alcohol use and tobacco are linked to breast cancer, and tobacco is the '
                                'single greatest avoidable risk factor for cancer deaths overall.'),
    ('⚖️', 'Weight & inactivity', 'Excess body weight and little physical activity are linked to several '
                                  'cancers, including breast cancer after menopause.'),
    ('☢️', 'Radiation exposure', 'Earlier radiation treatment to the chest area is a recognised risk factor.'),
    ('🔍', 'Often no clear cause', 'WHO notes roughly half of breast cancers occur in women with no specific '
                                   'risk factor beyond being female and getting older - so awareness helps everyone.'),
]

SIGNS = [
    ('🤲', 'A new lump or thickening', 'In the breast or underarm. Most lumps are not cancer, but always get them checked.'),
    ('📐', 'Change in size or shape', 'One breast looking different from before, or a change in the outline.'),
    ('🍊', 'Skin changes', 'Dimpling, puckering, redness or an "orange-peel" texture.'),
    ('💧', 'Nipple changes', 'Turning inwards, new discharge (especially bloody), crusting or soreness.'),
]

SELF_CHECK = [
    ('🪞', 'Look', 'Stand in front of a mirror, arms down and then raised. Look for changes in shape, skin or nipples.'),
    ('🤲', 'Feel', 'Use the pads of your fingers in small circles, covering the whole breast and up to the armpit.'),
    ('🛏️', 'Lie down', 'Lying flat spreads the tissue evenly, which makes small changes easier to notice.'),
    ('🚿', 'In the shower', 'Wet, soapy skin glides easily - a comfortable moment to check each side.'),
]

CARE = [
    ('🏃‍♀️', 'Keep moving', 'Aim for regular activity - WHO suggests at least 150 minutes of moderate exercise a week. '
                          'Dancing, brisk walking and cycling all count.'),
    ('🥗', 'Eat the rainbow', 'Plenty of fruit, vegetables, whole grains and pulses; go easy on ultra-processed food.'),
    ('🚭', 'Stay tobacco-free', 'Not smoking is one of the best things you can do for your health. Support is available if you want to quit.'),
    ('🍵', 'Limit alcohol', 'The less the better. Swap in sparkling water, tender coconut water or herbal tea.'),
    ('⚖️', 'Healthy weight', 'Small, steady habits beat crash diets. Celebrate progress, not perfection.'),
    ('🤱', 'Breastfeed if you can', 'Breastfeeding is linked with lower breast cancer risk and is great for the baby too.'),
    ('😴', 'Sleep & calm', 'Rest, hobbies, yoga, time with loved ones - looking after your mind helps your whole body.'),
    ('🩺', 'Know your normal & get screened', 'Check in with your body, and ask your doctor when screening is right for you. '
                                              'Finding cancer early gives the best chance of cure.'),
]

CHECKLIST = ['Moved my body for 30 minutes', 'Ate at least 5 fruits / vegetables', 'Drank water instead of sugary drinks',
             'Took a calm 5-minute breathing break', 'Got a good night\'s sleep', 'Checked in on my body / booked a check-up']


def _gallery():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'photos')
    files = []
    for ext in ('jpg', 'jpeg', 'png', 'webp'):
        files += glob.glob(os.path.join(root, f'*.{ext}'))
    return sorted(files)[:6]


def render_awareness_home(set_page=None):
    _md(AW_CSS)
    ribbon = _img(ribbon_svg(), "width:120px")
    _md(f'''<div class="aw-hero">{_floaters()}
      <div class="aw-tag">🎀 Breast Cancer Awareness</div>
      <div class="aw-ribbon" style="width:120px;margin:1rem auto 0">{ribbon}</div>
      <h1>Knowledge is <em>power</em>.<br>Early detection saves lives.</h1>
      <div class="aw-sub">Understand what causes breast cancer, learn the early signs, and discover simple,
      positive habits that care for your body and mind. You are not alone - and hope is real.</div>
      <div class="aw-chips">
        <div class="aw-chip"><b>670,000</b><span>deaths worldwide in 2022 (WHO)</span></div>
        <div class="aw-chip"><b>157 / 185</b><span>countries where it is the most common cancer in women</span></div>
        <div class="aw-chip"><b>30-50%</b><span>of cancers can be prevented (WHO)</span></div>
        <div class="aw-chip"><b>Early = hope</b><span>many cancers are curable when found early</span></div>
      </div></div>''')
    _md(_wave('#080C14'))
    if set_page is not None:
        _, b1, b2, _ = st.columns([2, 2, 2, 2])
        if b1.button("Try the AI screening assistant", key="cta_assess", use_container_width=True, type="primary"):
            set_page("Assessment")
        if b2.button("Support the cause 💗", key="cta_donate", use_container_width=True):
            set_page("Donate")

    # --- solidarity illustration
    _md(f'<div class="aw-sec" style="padding-top:1.2rem">'
        f'{_img(solidarity_svg(), "width:100%;max-width:900px;display:block;margin:0 auto;border-radius:22px;box-shadow:0 14px 40px #FF5C9A22")}'
        f'<div class="aw-p" style="margin-top:.8rem">Breast cancer affects people of every background, in every country. '
        f'Together we can spread awareness, support each other and fund better care. 🎀</div></div>')

    # --- causes
    _md('<div class="aw-sec"><div class="aw-h">What can cause breast cancer?</div>'
        '<div class="aw-p">There is rarely a single cause. Risk comes from a mix of age, genes, hormones and lifestyle. '
        'Having a risk factor does <b>not</b> mean you will get cancer - and many people with none still do.</div>'
        + _cards(CAUSES) + '</div>')

    # --- signs
    _md('<div class="aw-sec"><div class="aw-h">Know the early signs</div>'
        '<div class="aw-p">Noticing a change early is powerful. Most changes turn out to be harmless, '
        'but please see a doctor if you notice any of these.</div>' + _cards(SIGNS) +
        '<div class="aw-note">⚠️ This page is for education only and is not medical advice. '
        'If you notice a change, see a qualified doctor promptly.</div></div>')

    # --- self-check
    _md('<div class="aw-sec"><div class="aw-h">A gentle 4-step self-awareness routine</div>'
        '<div class="aw-p">Getting familiar with how your breasts normally look and feel helps you spot what is new. '
        'It does not replace screening or a doctor\'s examination.</div>' + _cards(SELF_CHECK) + '</div>')

    # --- positive care
    _md('<div class="aw-sec"><div class="aw-h">Care for yourself - the positive way 🌱</div>'
        '<div class="aw-p">Healthy habits will not guarantee protection, but they lower risk of many cancers and make you '
        'feel stronger. Start small - one good habit at a time.</div>' + _cards(CARE, 'g') + '</div>')

    _md('<div class="aw-hope"><q>Many cancers have a high chance of cure if diagnosed early and treated appropriately.</q>'
        '<small>- World Health Organization</small></div>')

    # --- interactive wellness checklist
    _md('<div class="aw-sec"><div class="aw-h">Your daily wellness checklist ✨</div>'
        '<div class="aw-p">Tick what you did today and watch your progress bloom.</div></div>')
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        done = sum(1 for i, item in enumerate(CHECKLIST) if st.checkbox(item, key=f'wellness_{i}'))
        st.progress(done / len(CHECKLIST), text=f"{done} of {len(CHECKLIST)} done today")
        if done == len(CHECKLIST):
            if not st.session_state.get('wellness_balloons'):
                st.balloons()
                st.session_state['wellness_balloons'] = True
            st.success("Amazing! Every one of these habits is a gift to your future self. 💗")
        else:
            st.session_state['wellness_balloons'] = False

    # --- optional photo gallery
    photos = _gallery()
    if photos:
        _md('<div class="aw-sec"><div class="aw-h">Moments of hope 📸</div></div>')
        cols = st.columns(min(3, len(photos)))
        for i, p in enumerate(photos):
            cols[i % len(cols)].image(p, use_container_width=True)

    _md('<div class="aw-sec" style="padding-bottom:.5rem"><div class="aw-p" style="font-size:.82rem">'
        'Sources: WHO fact sheets on '
        '<a class="aw-link" style="margin:0 .2rem" href="https://www.who.int/news-room/fact-sheets/detail/breast-cancer" target="_blank" rel="noopener">breast cancer</a>'
        'and <a class="aw-link" style="margin:0 .2rem" href="https://www.who.int/news-room/fact-sheets/detail/cancer" target="_blank" rel="noopener">cancer</a>; '
        'EU Knowledge4Policy breast cancer summary. Always follow your own doctor\'s advice.</div></div>')
    _md(f'<div class="aw-sec" style="padding-top:0"><div class="aw-h" style="font-size:1.6rem">Our AI screening assistant</div>'
        f'<div class="aw-p">Below is the multi-modal AI system built for this project.</div></div>')


# --------------------------------------------------------------------------
# RESOURCES
# --------------------------------------------------------------------------
RESOURCES = [
    ('🌍', 'World Health Organization (WHO)', 'Global fact sheets on breast cancer and cancer: causes, symptoms, screening and treatment.',
     [('Breast cancer fact sheet', 'https://www.who.int/news-room/fact-sheets/detail/breast-cancer'),
      ('Cancer fact sheet', 'https://www.who.int/news-room/fact-sheets/detail/cancer')]),
    ('🎗️', 'American Cancer Society', 'Plain-language guides, a 24/7 helpline (US: 1-800-227-2345) and patient support programmes.',
     [('Visit cancer.org', 'https://www.cancer.org')]),
    ('🔬', 'National Cancer Institute (US)', 'Research-based information on breast cancer types, treatment and clinical trials.',
     [('Visit cancer.gov', 'https://www.cancer.gov')]),
    ('💬', 'Breastcancer.org', 'Patient-friendly explanations, community forums and survivor stories.',
     [('Visit breastcancer.org', 'https://www.breastcancer.org')]),
    ('🇬🇧', 'Cancer Research UK', 'Clear, trusted information on cancer symptoms, screening and research.',
     [('Visit cancerresearchuk.org', 'https://www.cancerresearchuk.org')]),
    ('🇮🇳', 'Cancer Institute (WIA), Chennai', 'A non-profit comprehensive cancer centre in Adyar, Chennai - treatment, research and prevention.',
     [('Official website', 'https://cancerinstitutewia.in')]),
    ('🏥', 'Tata Memorial Centre, Mumbai', 'One of India\'s leading cancer care and research institutions.',
     [('Official website', 'https://tmc.gov.in')]),
]


def render_resources():
    _md(AW_CSS)
    _md(f'''<div class="aw-hero" style="padding-bottom:4.5rem">{_floaters()}
      <div class="aw-tag">📚 Trusted resources</div>
      <h1>Learn from <em>real experts</em></h1>
      <div class="aw-sub">Reliable organisations where you can read more, find support and get help.
      These are independent websites - this project is not affiliated with them.</div></div>''')
    _md(_wave('#080C14'))
    cards = ['<div class="aw-sec"><div class="aw-grid">']
    for icon, name, desc, links in RESOURCES:
        btns = ''.join(f'<a class="aw-link" href="{u}" target="_blank" rel="noopener">{t} ↗</a> ' for t, u in links)
        cards.append(f'<div class="aw-card"><div class="aw-ico">{icon}</div><h4>{name}</h4><p>{desc}</p>{btns}</div>')
    cards.append('</div>')
    cards.append('<div class="aw-note">If you or someone you know has symptoms, please contact a doctor or your local health '
                 'service. This website cannot diagnose or treat any condition.</div></div>')
    _md(''.join(cards))


# --------------------------------------------------------------------------
# DONATE
# --------------------------------------------------------------------------
CHARITIES = [
    ('🏥', 'Cancer Institute (WIA) - Adyar, Chennai',
     'A voluntary, non-profit cancer hospital and research centre in Chennai that depends largely on hospital income '
     'and public contributions to treat patients, including those who cannot afford care.',
     'https://cancerinstitutewia.in', 'Go to official site'),
    ('🇮🇳', 'Tata Memorial Centre - Mumbai',
     'India\'s pioneer cancer institute. Donations support patient welfare, infrastructure, research and blood-transfusion '
     'support for underprivileged patients.',
     'https://tmc.gov.in/m_donation', 'Donate on official page'),
    ('🎗️', 'American Cancer Society',
     'Funds research, advocacy and patient support worldwide, including its breast-cancer programmes. Gifts are '
     'tax-deductible in the US.',
     'https://www.cancer.org/donate', 'Donate on official page'),
]


def render_donate():
    _md(AW_CSS)
    ribbon = _img(ribbon_svg(), "width:90px")
    _md(f'''<div class="aw-hero" style="padding-bottom:4.5rem">{_floaters()}
      <div class="aw-tag">💗 Give hope</div>
      <div class="aw-ribbon" style="width:90px;margin:.8rem auto 0">{ribbon}</div>
      <h1>Every gift gives <em>someone hope</em></h1>
      <div class="aw-sub">Cancer treatment can be life-saving and very expensive. Support trusted cancer hospitals and charities -
      every contribution, big or small, helps fund treatment, research and early detection.</div></div>''')
    _md(_wave('#080C14'))

    cards = ['<div class="aw-sec"><div class="aw-h">Where you can donate</div>'
             '<div class="aw-p">These are real, established organisations. You will be taken to <b>their own official website</b> '
             'to give - this app never handles any payment.</div><div class="aw-grid">']
    for icon, name, desc, url, label in CHARITIES:
        cards.append(f'<div class="aw-card aw-tier"><div class="aw-ico">{icon}</div><h4>{name}</h4><p>{desc}</p>'
                     f'<a class="aw-link" href="{url}" target="_blank" rel="noopener">{label} ↗</a></div>')
    cards.append('</div>')
    cards.append('<div class="aw-note">🔒 <b>Give safely:</b> type the website address yourself or use the links above, check for '
                 'https and the organisation\'s name, never share card/UPI PINs or OTPs, and ask for an official receipt. '
                 'Tax benefits depend on your country - confirm with the organisation. '
                 'Beware of unofficial QR codes and messages asking for urgent transfers.</div></div>')
    _md(''.join(cards))

    _md('<div class="aw-sec"><div class="aw-h">Other ways to help - free 🌸</div>'
        '<div class="aw-p">Not able to donate money? You can still make a real difference.</div>'
        + _cards([('📣', 'Spread awareness', 'Share this page and the early-signs checklist with family and friends.'),
                  ('🤝', 'Volunteer', 'Join awareness walks, support groups or hospital volunteering programmes near you.'),
                  ('💇‍♀️', 'Donate hair', 'Some cancer centres accept hair donations for wigs for patients - ask your local centre.'),
                  ('💌', 'Be there', 'A call, a meal or a ride to an appointment means more than you think.')], 'g') + '</div>')

    # --- symbolic pledge wall (no money involved)
    _md('<div class="aw-sec"><div class="aw-h">Light a ribbon of hope 🎀</div>'
        '<div class="aw-p">A symbolic pledge wall - leave a message of support. '
        '<b>No payment is taken and nothing is stored beyond this browser session.</b></div></div>')
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        name = st.text_input("Your first name (optional)", max_chars=24, key='pledge_name')
        msg = st.text_input("A message of hope", max_chars=80, key='pledge_msg',
                            placeholder="Stay strong - we are with you!")
        if st.button("Light a ribbon 🎀", use_container_width=True, key='pledge_btn'):
            wall = st.session_state.setdefault('pledges', [])
            wall.insert(0, f"{(name or 'Someone').strip()}: {(msg or 'Sending love and strength').strip()}")
            del wall[30:]
            st.balloons()
    wall = st.session_state.get('pledges', [])
    if wall:
        chips = ''.join(f'<span class="aw-pledge">🎀 {_html.escape(p)}</span>' for p in wall)
        _md(f'<div class="aw-sec" style="text-align:center"><div class="aw-p" style="margin-bottom:.5rem">'
            f'{len(wall)} ribbon(s) lit this session</div>{chips}</div>')
    _md('<div class="aw-sec"><div class="aw-note">This student project is not a charity and is not affiliated with any '
        'organisation listed here. Please verify any organisation yourself before donating.</div></div>')
