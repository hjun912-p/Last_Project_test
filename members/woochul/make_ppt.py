from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

# ── 색상 팔레트 ──
C_BG       = RGBColor(0x0D, 0x1B, 0x2A)   # 진한 네이비
C_CARD     = RGBColor(0x1A, 0x2E, 0x44)   # 카드 배경
C_ACCENT   = RGBColor(0x00, 0xC8, 0xFF)   # 시안 포인트
C_ACCENT2  = RGBColor(0x7C, 0x3A, 0xFF)   # 보라 포인트
C_GREEN    = RGBColor(0x00, 0xE5, 0x96)   # 그린 포인트
C_ORANGE   = RGBColor(0xFF, 0x8C, 0x00)   # 오렌지
C_WHITE    = RGBColor(0xFF, 0xFF, 0xFF)
C_GRAY     = RGBColor(0xA0, 0xB0, 0xC0)
C_CODE_BG  = RGBColor(0x05, 0x0F, 0x19)   # 코드블록 배경

W = float(prs.slide_width)
H = float(prs.slide_height)

blank_layout = prs.slide_layouts[6]

# ══════════════════════════════════════════════
# 헬퍼 함수
# ══════════════════════════════════════════════

def add_rect(slide, x, y, w, h, fill_color=None, alpha=None, line_color=None, line_width=Pt(0)):
    shape = slide.shapes.add_shape(1, Emu(x), Emu(y), Emu(w), Emu(h))
    shape.line.width = line_width
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = line_width
    else:
        shape.line.fill.background()
    return shape

def add_text_box(slide, text, x, y, w, h,
                 font_size=Pt(18), bold=False, color=C_WHITE,
                 align=PP_ALIGN.LEFT, wrap=True, italic=False):
    txBox = slide.shapes.add_textbox(Emu(x), Emu(y), Emu(w), Emu(h))
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = font_size
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox

def add_bg(slide):
    add_rect(slide, 0, 0, W, H, fill_color=C_BG)

def inches(n):
    return Inches(n)

def px(n):
    return Emu(n * 9144)

# ══════════════════════════════════════════════
# 슬라이드 1 — 표지
# ══════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
add_bg(slide)

# 상단 라인
add_rect(slide, 0, Inches(0.3), W, Pt(3).emu, fill_color=C_ACCENT)

# 태그
tag = add_rect(slide, Inches(1.2), Inches(1.6), Inches(3.5), Inches(0.45), fill_color=C_ACCENT)
add_text_box(slide, "팀 개발 입문 가이드", Inches(1.25), Inches(1.63), Inches(3.4), Inches(0.4),
             font_size=Pt(13), bold=True, color=C_BG, align=PP_ALIGN.LEFT)

# 메인 타이틀
add_text_box(slide, "GitHub & 개발환경", Inches(1.2), Inches(2.15), Inches(10), Inches(1.2),
             font_size=Pt(52), bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
add_text_box(slide, "완전 정복 튜토리얼", Inches(1.2), Inches(3.2), Inches(10), Inches(1.0),
             font_size=Pt(44), bold=True, color=C_ACCENT, align=PP_ALIGN.LEFT)

# 서브타이틀
add_text_box(slide,
    "Git 용어 · 터미널 명령어 · AI로 GitHub 명령 · 프로젝트 환경 구성",
    Inches(1.2), Inches(4.35), Inches(10), Inches(0.5),
    font_size=Pt(16), color=C_GRAY, align=PP_ALIGN.LEFT)

# 하단 라인
add_rect(slide, 0, H - Inches(0.55), W, Pt(2).emu, fill_color=C_ACCENT2)
add_text_box(slide, "InSIGHT Project  |  hnf Team", Inches(1.2), H - Inches(0.5), Inches(8), Inches(0.4),
             font_size=Pt(13), color=C_GRAY, align=PP_ALIGN.LEFT)

# ══════════════════════════════════════════════
# 슬라이드 2 — 목차
# ══════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
add_bg(slide)
add_rect(slide, 0, 0, Inches(0.18), H, fill_color=C_ACCENT)
add_text_box(slide, "INDEX", Inches(0.4), Inches(0.3), Inches(4), Inches(0.6),
             font_size=Pt(13), bold=True, color=C_ACCENT, align=PP_ALIGN.LEFT)
add_text_box(slide, "목차", Inches(0.4), Inches(0.8), Inches(12), Inches(0.8),
             font_size=Pt(38), bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)

items = [
    ("01", "GitHub 핵심 용어 사전",        "커밋·푸쉬·풀 등 꼭 알아야 할 단어 정리",          C_ACCENT),
    ("02", "터미널 Git 명령어 치트시트",    "실제로 쓰는 명령어만 골라 한눈에",               C_ACCENT2),
    ("03", "AI로 GitHub 명령하기",         "Claude / Gemini CLI 에게 채팅으로 Git 시키기",    C_GREEN),
    ("04", "프로젝트 개발환경 구성",        "Miniconda → hnf 환경 → 앱 실행까지",            C_ORANGE),
]

for i, (num, title, sub, col) in enumerate(items):
    y = Inches(1.9) + i * Inches(1.2)
    add_rect(slide, Inches(0.4), y, Inches(12.5), Inches(1.05), fill_color=C_CARD,
             line_color=col, line_width=Pt(1.5))
    add_rect(slide, Inches(0.4), y, Inches(0.85), Inches(1.05), fill_color=col)
    add_text_box(slide, num, Inches(0.4), y + Inches(0.22), Inches(0.85), Inches(0.6),
                 font_size=Pt(22), bold=True, color=C_BG, align=PP_ALIGN.CENTER)
    add_text_box(slide, title, Inches(1.4), y + Inches(0.08), Inches(7), Inches(0.5),
                 font_size=Pt(20), bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
    add_text_box(slide, sub, Inches(1.4), y + Inches(0.55), Inches(9), Inches(0.4),
                 font_size=Pt(14), color=C_GRAY, align=PP_ALIGN.LEFT)

# ══════════════════════════════════════════════
# 슬라이드 3 — GitHub 핵심 용어 (1/2)
# ══════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
add_bg(slide)
add_rect(slide, 0, 0, Inches(0.18), H, fill_color=C_ACCENT)
add_text_box(slide, "01  GitHub 핵심 용어 사전", Inches(0.4), Inches(0.25), Inches(12), Inches(0.55),
             font_size=Pt(13), bold=True, color=C_ACCENT)
add_text_box(slide, "꼭 알아야 할 Git 단어들", Inches(0.4), Inches(0.72), Inches(12), Inches(0.7),
             font_size=Pt(32), bold=True, color=C_WHITE)

terms = [
    ("Repository\n(저장소)",  "프로젝트의 모든 파일과 변경 이력이 저장되는 공간.\n쉽게 말해 '프로젝트 폴더의 클라우드 버전'",          "📦", C_ACCENT),
    ("Commit\n(커밋)",        "변경 내용을 로컬에 저장하는 행위.\n사진 찍듯 현재 상태를 기록해 두는 것",                          "📸", C_ACCENT2),
    ("Push\n(푸쉬)",          "로컬에 커밋한 내용을 GitHub 서버(원격)로 업로드.\n'내 PC → GitHub 서버'",                         "⬆️", C_GREEN),
    ("Pull\n(풀)",            "GitHub 서버의 최신 내용을 내 PC로 다운로드.\n'GitHub 서버 → 내 PC'",                             "⬇️", C_ORANGE),
    ("Clone\n(클론)",         "GitHub에 있는 저장소를 내 PC로 복사해 오는 것.\n처음 한 번만 하면 됨",                            "📋", C_ACCENT),
    ("Branch\n(브랜치)",      "메인 코드를 건드리지 않고 기능을 개발하는\n독립된 작업 공간. 나뭇가지처럼 갈라짐",                   "🌿", C_ACCENT2),
]

cols = 3
for i, (term, desc, icon, col) in enumerate(terms):
    row = i // cols
    col_idx = i % cols
    x = Inches(0.4) + col_idx * Inches(4.28)
    y = Inches(1.55) + row * Inches(2.55)
    add_rect(slide, x, y, Inches(4.1), Inches(2.38), fill_color=C_CARD,
             line_color=col, line_width=Pt(1.5))
    add_text_box(slide, icon, x + Inches(0.18), y + Inches(0.12), Inches(0.7), Inches(0.5),
                 font_size=Pt(22), align=PP_ALIGN.LEFT)
    add_text_box(slide, term, x + Inches(0.18), y + Inches(0.5), Inches(3.7), Inches(0.75),
                 font_size=Pt(17), bold=True, color=col, align=PP_ALIGN.LEFT)
    add_text_box(slide, desc, x + Inches(0.18), y + Inches(1.22), Inches(3.7), Inches(1.05),
                 font_size=Pt(12.5), color=C_GRAY, align=PP_ALIGN.LEFT)

# ══════════════════════════════════════════════
# 슬라이드 4 — GitHub 핵심 용어 (2/2)
# ══════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
add_bg(slide)
add_rect(slide, 0, 0, Inches(0.18), H, fill_color=C_ACCENT)
add_text_box(slide, "01  GitHub 핵심 용어 사전", Inches(0.4), Inches(0.25), Inches(12), Inches(0.55),
             font_size=Pt(13), bold=True, color=C_ACCENT)
add_text_box(slide, "Git 흐름 한눈에 보기", Inches(0.4), Inches(0.72), Inches(12), Inches(0.7),
             font_size=Pt(32), bold=True, color=C_WHITE)

# 흐름도
flow = [
    ("내 PC\n(Local)", C_ACCENT2),
    ("Staging\nArea",  C_ACCENT),
    ("Local\nRepo",    C_GREEN),
    ("GitHub\n(Remote)", C_ORANGE),
]
arrows = ["git add →", "git commit →", "git push →"]
box_w = Inches(2.4)
box_h = Inches(1.4)
start_x = Inches(0.5)
y_box = Inches(1.65)
arrow_w = Inches(1.15)

for i, (label, col) in enumerate(flow):
    x = start_x + i * (box_w + arrow_w)
    add_rect(slide, x, y_box, box_w, box_h, fill_color=col)
    add_text_box(slide, label, x, y_box + Inches(0.3), box_w, Inches(0.85),
                 font_size=Pt(16), bold=True, color=C_BG, align=PP_ALIGN.CENTER)
    if i < 3:
        ax = x + box_w
        add_rect(slide, ax, y_box + Inches(0.55), arrow_w, Pt(3).emu, fill_color=C_GRAY)
        add_text_box(slide, arrows[i], ax, y_box + Inches(0.62), arrow_w, Inches(0.4),
                     font_size=Pt(10), color=C_GRAY, align=PP_ALIGN.CENTER)

# 역방향 pull
add_rect(slide, start_x, y_box + Inches(1.65), Inches(12.0), Pt(2).emu, fill_color=C_ORANGE)
add_text_box(slide, "← git pull  (GitHub → 내 PC 최신화)",
             start_x, y_box + Inches(1.78), Inches(12), Inches(0.45),
             font_size=Pt(14), color=C_ORANGE, align=PP_ALIGN.CENTER)

# 추가 용어
extra = [
    ("Merge", "두 브랜치를 하나로 합치는 작업", C_ACCENT),
    ("Pull Request (PR)", "내 브랜치를 main에 합쳐달라고 요청하는 것. 팀원 코드 리뷰 진행", C_ACCENT2),
    ("Conflict", "두 사람이 같은 줄을 다르게 수정했을 때 발생하는 충돌. 직접 선택해서 해결해야 함", C_GREEN),
    (".gitignore", "GitHub에 올리면 안 되는 파일 목록 (.env, API 키 등) 을 적어두는 파일", C_ORANGE),
]

for i, (term, desc, col) in enumerate(extra):
    x = Inches(0.4) + (i % 2) * Inches(6.45)
    y = Inches(3.6) + (i // 2) * Inches(1.55)
    add_rect(slide, x, y, Inches(6.2), Inches(1.38), fill_color=C_CARD,
             line_color=col, line_width=Pt(1.5))
    add_rect(slide, x, y, Inches(0.12), Inches(1.38), fill_color=col)
    add_text_box(slide, term, x + Inches(0.25), y + Inches(0.1), Inches(5.8), Inches(0.45),
                 font_size=Pt(16), bold=True, color=col)
    add_text_box(slide, desc, x + Inches(0.25), y + Inches(0.55), Inches(5.8), Inches(0.75),
                 font_size=Pt(13), color=C_GRAY)

# ══════════════════════════════════════════════
# 슬라이드 5 — 터미널 Git 명령어 치트시트
# ══════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
add_bg(slide)
add_rect(slide, 0, 0, Inches(0.18), H, fill_color=C_ACCENT2)
add_text_box(slide, "02  터미널 Git 명령어 치트시트", Inches(0.4), Inches(0.25), Inches(12), Inches(0.55),
             font_size=Pt(13), bold=True, color=C_ACCENT2)
add_text_box(slide, "자주 쓰는 명령어만 골라 정리했어요", Inches(0.4), Inches(0.72), Inches(12), Inches(0.7),
             font_size=Pt(32), bold=True, color=C_WHITE)

sections = [
    ("기본 설정 (처음 한 번)", C_ACCENT, [
        ("git config --global user.name \"이름\"",   "내 이름 등록"),
        ("git config --global user.email \"이메일\"", "내 이메일 등록"),
        ("git clone 저장소_주소",                    "GitHub 프로젝트를 내 PC로 복사"),
    ]),
    ("매일 쓰는 명령어", C_GREEN, [
        ("git pull origin main",  "GitHub → 내 PC 최신화 (작업 시작 전 필수!)"),
        ("git add .",             "변경된 모든 파일 한번에 스테이징"),
        ("git commit -m \"메시지\"", "스테이징 내용을 로컬에 저장"),
        ("git push origin main",  "내 커밋을 GitHub에 업로드"),
    ]),
    ("확인 명령어", C_ORANGE, [
        ("git status",   "현재 변경사항 목록 확인"),
        ("git log",      "커밋 이력 확인"),
        ("git diff",     "변경된 내용 상세 보기"),
        ("git branch",   "현재 브랜치 확인"),
    ]),
]

col_x = [Inches(0.4), Inches(4.65), Inches(8.9)]
col_w = Inches(4.1)

for i, (title, col, cmds) in enumerate(sections):
    x = col_x[i]
    y_start = Inches(1.55)
    add_rect(slide, x, y_start, col_w, Inches(0.45), fill_color=col)
    add_text_box(slide, title, x + Inches(0.15), y_start + Inches(0.07), col_w - Inches(0.2), Inches(0.35),
                 font_size=Pt(14), bold=True, color=C_BG)
    y = y_start + Inches(0.55)
    for cmd, desc in cmds:
        add_rect(slide, x, y, col_w, Inches(0.88), fill_color=C_CODE_BG,
                 line_color=col, line_width=Pt(0.8))
        add_text_box(slide, cmd, x + Inches(0.12), y + Inches(0.04), col_w - Inches(0.15), Inches(0.42),
                     font_size=Pt(11.5), bold=True, color=col, align=PP_ALIGN.LEFT)
        add_text_box(slide, desc, x + Inches(0.12), y + Inches(0.44), col_w - Inches(0.15), Inches(0.38),
                     font_size=Pt(11), color=C_GRAY, align=PP_ALIGN.LEFT)
        y += Inches(0.96)

# 작업 순서 TIP
tip_y = Inches(6.15)
add_rect(slide, Inches(0.4), tip_y, Inches(12.5), Inches(0.78), fill_color=C_CARD,
         line_color=C_ACCENT, line_width=Pt(1.5))
add_text_box(slide, "💡 매일 작업 순서",
             Inches(0.6), tip_y + Inches(0.08), Inches(2.5), Inches(0.35),
             font_size=Pt(13), bold=True, color=C_ACCENT)
add_text_box(slide,
    "① git pull origin main  →  ② 코드 수정  →  ③ git add .  →  ④ git commit -m \"메시지\"  →  ⑤ git push origin main",
    Inches(3.0), tip_y + Inches(0.08), Inches(9.7), Inches(0.55),
    font_size=Pt(13), color=C_WHITE)

# ══════════════════════════════════════════════
# 슬라이드 6 — AI로 GitHub 명령하기 (개요)
# ══════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
add_bg(slide)
add_rect(slide, 0, 0, Inches(0.18), H, fill_color=C_GREEN)
add_text_box(slide, "03  AI로 GitHub 명령하기", Inches(0.4), Inches(0.25), Inches(12), Inches(0.55),
             font_size=Pt(13), bold=True, color=C_GREEN)
add_text_box(slide, "채팅으로 Git 시키기", Inches(0.4), Inches(0.72), Inches(12), Inches(0.7),
             font_size=Pt(32), bold=True, color=C_WHITE)

# 설명
add_text_box(slide,
    "터미널 명령어 외우기 어렵다면?  AI에게 말로 시키면 됩니다.\nClaude Code CLI 또는 Gemini CLI에게 GitHub 권한을 주면 커밋·푸쉬·풀을 채팅 한 줄로!",
    Inches(0.4), Inches(1.52), Inches(12.5), Inches(0.75),
    font_size=Pt(15), color=C_GRAY)

# 두 개 카드
cards = [
    ("Claude Code CLI", "Anthropic 공식 AI 코딩 어시스턴트\n터미널에서 대화하듯 코드·Git 작업 가능", C_ACCENT2,
     ["$ npm install -g @anthropic-ai/claude-code", "$ claude  (프로젝트 폴더에서 실행)", "→ GitHub 토큰 연동 후 바로 사용"]),
    ("Gemini CLI", "Google DeepMind의 터미널용 AI\n무료 티어 제공, Google 계정으로 바로 사용 가능", C_GREEN,
     ["$ npm install -g @google/gemini-cli", "$ gemini  (프로젝트 폴더에서 실행)", "→ Google 계정 로그인 후 바로 사용"]),
]

for i, (name, desc, col, cmds) in enumerate(cards):
    x = Inches(0.4) + i * Inches(6.5)
    y = Inches(2.45)
    add_rect(slide, x, y, Inches(6.2), Inches(4.75), fill_color=C_CARD,
             line_color=col, line_width=Pt(2))
    add_rect(slide, x, y, Inches(6.2), Inches(0.5), fill_color=col)
    add_text_box(slide, name, x + Inches(0.18), y + Inches(0.08), Inches(5.8), Inches(0.38),
                 font_size=Pt(17), bold=True, color=C_BG)
    add_text_box(slide, desc, x + Inches(0.18), y + Inches(0.62), Inches(5.8), Inches(0.75),
                 font_size=Pt(13), color=C_GRAY)
    cy = y + Inches(1.52)
    for cmd in cmds:
        add_rect(slide, x + Inches(0.18), cy, Inches(5.82), Inches(0.62), fill_color=C_CODE_BG)
        add_text_box(slide, cmd, x + Inches(0.32), cy + Inches(0.1), Inches(5.55), Inches(0.45),
                     font_size=Pt(11.5), bold=True, color=col)
        cy += Inches(0.72)

# ══════════════════════════════════════════════
# 슬라이드 7 — Claude Code 권한 설정 & 채팅 예시
# ══════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
add_bg(slide)
add_rect(slide, 0, 0, Inches(0.18), H, fill_color=C_GREEN)
add_text_box(slide, "03  AI로 GitHub 명령하기 — Claude Code 실전", Inches(0.4), Inches(0.25), Inches(12), Inches(0.55),
             font_size=Pt(13), bold=True, color=C_GREEN)
add_text_box(slide, "Claude Code 설치 → 권한 설정 → 채팅으로 명령", Inches(0.4), Inches(0.72), Inches(12), Inches(0.7),
             font_size=Pt(28), bold=True, color=C_WHITE)

# 왼쪽: 설치 & 권한
left_steps = [
    ("STEP 1", "Node.js 설치 확인", "node --version  # v18 이상 필요", C_ACCENT),
    ("STEP 2", "Claude Code 설치", "npm install -g @anthropic-ai/claude-code", C_ACCENT2),
    ("STEP 3", "프로젝트 폴더에서 실행", "cd ~/github/Last_Project_test\nclaude", C_GREEN),
    ("STEP 4", "GitHub 권한 부여", "claude 실행 후 안내에 따라\nGitHub 토큰 입력 또는 OAuth 로그인", C_ORANGE),
]

y = Inches(1.58)
for step, title, cmd, col in left_steps:
    add_rect(slide, Inches(0.4), y, Inches(5.8), Inches(1.15), fill_color=C_CARD,
             line_color=col, line_width=Pt(1))
    add_rect(slide, Inches(0.4), y, Inches(0.85), Inches(1.15), fill_color=col)
    add_text_box(slide, step, Inches(0.4), y + Inches(0.35), Inches(0.85), Inches(0.42),
                 font_size=Pt(10), bold=True, color=C_BG, align=PP_ALIGN.CENTER)
    add_text_box(slide, title, Inches(1.35), y + Inches(0.06), Inches(4.7), Inches(0.4),
                 font_size=Pt(14), bold=True, color=col)
    add_text_box(slide, cmd, Inches(1.35), y + Inches(0.48), Inches(4.7), Inches(0.58),
                 font_size=Pt(11), color=C_GRAY)
    y += Inches(1.25)

# 오른쪽: 채팅 예시
add_text_box(slide, "채팅 명령 예시", Inches(6.6), Inches(1.58), Inches(6.5), Inches(0.45),
             font_size=Pt(16), bold=True, color=C_GREEN)

chats = [
    ("나 →", "방금 수정한 app.py 커밋하고 푸쉬해줘",                     True,  C_ACCENT2),
    ("AI  →", "app.py 변경사항을 확인했습니다.\ngit add app.py → git commit → git push 할게요.", False, C_GREEN),
    ("나 →", "팀원들이 올린 최신 코드 받아와줘",                         True,  C_ACCENT2),
    ("AI  →", "git pull origin main 실행합니다.\n최신 커밋 3개 반영됐습니다.",               False, C_GREEN),
    ("나 →", "어제 이후로 바뀐 파일 목록 보여줘",                         True,  C_ACCENT2),
    ("AI  →", "git diff HEAD~1 결과:\napp.py, requirements_app.txt 수정됨.",         False, C_GREEN),
]

cy = Inches(2.1)
for speaker, msg, is_user, col in chats:
    bx = Inches(6.6) if is_user else Inches(7.0)
    bw = Inches(6.1)
    bh = Inches(0.72) if "\n" not in msg else Inches(0.88)
    add_rect(slide, bx, cy, bw, bh, fill_color=C_CARD, line_color=col, line_width=Pt(0.8))
    add_text_box(slide, speaker, bx + Inches(0.1), cy + Inches(0.06), Inches(0.6), Inches(0.3),
                 font_size=Pt(10), bold=True, color=col)
    add_text_box(slide, msg, bx + Inches(0.75), cy + Inches(0.06), bw - Inches(0.85), bh - Inches(0.1),
                 font_size=Pt(11.5), color=C_WHITE)
    cy += bh + Inches(0.1)

# ══════════════════════════════════════════════
# 슬라이드 8 — Gemini CLI 실전
# ══════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
add_bg(slide)
add_rect(slide, 0, 0, Inches(0.18), H, fill_color=C_GREEN)
add_text_box(slide, "03  AI로 GitHub 명령하기 — Gemini CLI 실전", Inches(0.4), Inches(0.25), Inches(12), Inches(0.55),
             font_size=Pt(13), bold=True, color=C_GREEN)
add_text_box(slide, "Gemini CLI 설치 → 로그인 → 채팅으로 명령", Inches(0.4), Inches(0.72), Inches(12), Inches(0.7),
             font_size=Pt(28), bold=True, color=C_WHITE)

gemini_steps = [
    ("STEP 1", "Node.js 설치 확인",      "node --version  # v18 이상",             C_ACCENT),
    ("STEP 2", "Gemini CLI 설치",         "npm install -g @google/gemini-cli",      C_GREEN),
    ("STEP 3", "프로젝트 폴더에서 실행",  "cd ~/github/Last_Project_test\ngemini",  C_ACCENT2),
    ("STEP 4", "Google 계정 로그인",      "브라우저 자동 실행 → Google 계정 로그인\n무료 티어로 바로 사용 가능!", C_ORANGE),
]

y = Inches(1.58)
for step, title, cmd, col in gemini_steps:
    add_rect(slide, Inches(0.4), y, Inches(5.8), Inches(1.18), fill_color=C_CARD,
             line_color=col, line_width=Pt(1))
    add_rect(slide, Inches(0.4), y, Inches(0.85), Inches(1.18), fill_color=col)
    add_text_box(slide, step, Inches(0.4), y + Inches(0.38), Inches(0.85), Inches(0.42),
                 font_size=Pt(10), bold=True, color=C_BG, align=PP_ALIGN.CENTER)
    add_text_box(slide, title, Inches(1.35), y + Inches(0.08), Inches(4.7), Inches(0.4),
                 font_size=Pt(14), bold=True, color=col)
    add_text_box(slide, cmd, Inches(1.35), y + Inches(0.5), Inches(4.7), Inches(0.6),
                 font_size=Pt(11), color=C_GRAY)
    y += Inches(1.28)

# 오른쪽 채팅
add_text_box(slide, "Gemini CLI 채팅 예시", Inches(6.6), Inches(1.58), Inches(6.5), Inches(0.45),
             font_size=Pt(16), bold=True, color=C_GREEN)

gemini_chats = [
    ("나 →",     "현재 수정한 파일 전부 커밋하고 푸쉬해줘\n메시지는 'feat: 기능 추가'로",             True,  C_ACCENT2),
    ("Gemini →", "git add . && git commit -m 'feat: 기능 추가'\ngit push origin main 실행할게요.", False, C_GREEN),
    ("나 →",     "main 브랜치 최신 코드 받아와줘",                                                  True,  C_ACCENT2),
    ("Gemini →", "git pull origin main 완료.\n변경된 파일: app.py, environment_app.yml",            False, C_GREEN),
    ("나 →",     "지금 내 수정사항 상태 알려줘",                                                    True,  C_ACCENT2),
    ("Gemini →", "git status 결과:\nmodified: app.py (아직 커밋 안 됨)",                           False, C_GREEN),
]

cy = Inches(2.1)
for speaker, msg, is_user, col in gemini_chats:
    bx = Inches(6.6) if is_user else Inches(7.0)
    bw = Inches(6.1)
    bh = Inches(0.78) if "\n" in msg else Inches(0.65)
    add_rect(slide, bx, cy, bw, bh, fill_color=C_CARD, line_color=col, line_width=Pt(0.8))
    add_text_box(slide, speaker, bx + Inches(0.1), cy + Inches(0.05), Inches(0.75), Inches(0.3),
                 font_size=Pt(10), bold=True, color=col)
    add_text_box(slide, msg, bx + Inches(0.85), cy + Inches(0.05), bw - Inches(0.95), bh - Inches(0.08),
                 font_size=Pt(11.5), color=C_WHITE)
    cy += bh + Inches(0.1)

# ══════════════════════════════════════════════
# 슬라이드 9 — 개발환경 구성 (전체 흐름)
# ══════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
add_bg(slide)
add_rect(slide, 0, 0, Inches(0.18), H, fill_color=C_ORANGE)
add_text_box(slide, "04  프로젝트 개발환경 구성", Inches(0.4), Inches(0.25), Inches(12), Inches(0.55),
             font_size=Pt(13), bold=True, color=C_ORANGE)
add_text_box(slide, "InSIGHT 앱 로컬 실행까지 전체 흐름", Inches(0.4), Inches(0.72), Inches(12), Inches(0.7),
             font_size=Pt(32), bold=True, color=C_WHITE)

flow_steps = [
    ("01", "Miniconda\n설치 확인",         "conda --version",                             C_ACCENT),
    ("02", "프로젝트\nGit Pull",           "git pull origin main",                        C_ACCENT2),
    ("03", "hnf 환경\n생성",               "conda env create\n-f environment_app.yml",    C_GREEN),
    ("04", "환경\n활성화",                 "conda activate hnf",                          C_ORANGE),
    ("05", ".env 파일\n설정",              "GEMINI_API_KEY=\n...",                        C_ACCENT),
    ("06", "앱\n실행",                     "python app.py",                               C_ACCENT2),
]

step_w = Inches(1.95)
step_h = Inches(2.5)
start_x = Inches(0.4)
y_step  = Inches(1.58)

for i, (num, label, cmd, col) in enumerate(flow_steps):
    x = start_x + i * (step_w + Inches(0.25))
    # 화살표
    if i > 0:
        ax = x - Inches(0.25)
        add_rect(slide, ax, y_step + Inches(1.0), Inches(0.25), Pt(2).emu, fill_color=C_GRAY)
        add_text_box(slide, "▶", ax, y_step + Inches(0.82), Inches(0.25), Inches(0.35),
                     font_size=Pt(12), color=C_GRAY, align=PP_ALIGN.CENTER)
    # 박스
    add_rect(slide, x, y_step, step_w, step_h, fill_color=C_CARD,
             line_color=col, line_width=Pt(2))
    add_rect(slide, x, y_step, step_w, Inches(0.5), fill_color=col)
    add_text_box(slide, num, x, y_step + Inches(0.07), step_w, Inches(0.38),
                 font_size=Pt(16), bold=True, color=C_BG, align=PP_ALIGN.CENTER)
    add_text_box(slide, label, x, y_step + Inches(0.6), step_w, Inches(0.85),
                 font_size=Pt(14), bold=True, color=col, align=PP_ALIGN.CENTER)
    add_rect(slide, x + Inches(0.12), y_step + Inches(1.58), step_w - Inches(0.24), Inches(0.78),
             fill_color=C_CODE_BG)
    add_text_box(slide, cmd, x + Inches(0.18), y_step + Inches(1.62), step_w - Inches(0.3), Inches(0.72),
                 font_size=Pt(10.5), bold=True, color=col, align=PP_ALIGN.CENTER)

# 완료 메시지
add_rect(slide, Inches(0.4), Inches(4.32), Inches(12.5), Inches(0.68), fill_color=C_GREEN)
add_text_box(slide, "🎉  http://127.0.0.1:7860  브라우저에서 열리면 성공!",
             Inches(0.6), Inches(4.4), Inches(12), Inches(0.5),
             font_size=Pt(18), bold=True, color=C_BG, align=PP_ALIGN.CENTER)

# 라이브러리 요약
lib_y = Inches(5.18)
add_text_box(slide, "설치되는 주요 라이브러리", Inches(0.4), lib_y, Inches(12), Inches(0.4),
             font_size=Pt(14), bold=True, color=C_GRAY)

libs = [
    ("gradio", "웹 UI", C_ACCENT),
    ("opencv / Pillow", "이미지 처리", C_ACCENT2),
    ("PyWavelets / scipy", "AI 판별 분석", C_GREEN),
    ("google-genai", "Gemini API", C_ORANGE),
    ("instaloader / yt-dlp", "SNS 다운로드", C_ACCENT),
    ("python-dotenv", "API 키 관리", C_ACCENT2),
]

lx = Inches(0.4)
for lib, role, col in libs:
    lw = Inches(2.02)
    add_rect(slide, lx, lib_y + Inches(0.48), lw, Inches(0.78), fill_color=C_CARD,
             line_color=col, line_width=Pt(1))
    add_text_box(slide, lib,  lx + Inches(0.1), lib_y + Inches(0.52), lw - Inches(0.12), Inches(0.35),
                 font_size=Pt(11), bold=True, color=col)
    add_text_box(slide, role, lx + Inches(0.1), lib_y + Inches(0.83), lw - Inches(0.12), Inches(0.3),
                 font_size=Pt(10), color=C_GRAY)
    lx += lw + Inches(0.1)

# ══════════════════════════════════════════════
# 슬라이드 10 — 환경 구성 상세 명령어
# ══════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
add_bg(slide)
add_rect(slide, 0, 0, Inches(0.18), H, fill_color=C_ORANGE)
add_text_box(slide, "04  프로젝트 개발환경 구성 — 상세 명령어", Inches(0.4), Inches(0.25), Inches(12), Inches(0.55),
             font_size=Pt(13), bold=True, color=C_ORANGE)
add_text_box(slide, "터미널에 그대로 복사해서 쓰세요", Inches(0.4), Inches(0.72), Inches(12), Inches(0.7),
             font_size=Pt(32), bold=True, color=C_WHITE)

detail_steps = [
    ("① Miniconda 확인", C_ACCENT,
     "conda --version",
     "버전 숫자가 나오면 OK. 아무것도 안 뜨면 miniconda.com에서 설치"),
    ("② 프로젝트 최신화", C_ACCENT2,
     "cd ~/github/Last_Project_test\ngit pull origin main",
     "프로젝트 폴더로 이동 후 최신 코드 받기"),
    ("③ hnf 환경 생성 (처음 한 번만)", C_GREEN,
     "conda env create -f environment_app.yml",
     "3~10분 소요. 완료 후 .env 파일에\nGEMINI_API_KEY 등 API 키 설정"),
    ("④ 환경 활성화 & 앱 실행", C_ORANGE,
     "conda activate hnf\npython app.py",
     "프롬프트에 (hnf) 붙으면 성공. 이후 앱 실행"),
]

y = Inches(1.58)
for title, col, cmd, desc in detail_steps:
    add_rect(slide, Inches(0.4), y, Inches(12.5), Inches(1.32), fill_color=C_CARD,
             line_color=col, line_width=Pt(1.5))
    add_rect(slide, Inches(0.4), y, Inches(2.6), Inches(1.32), fill_color=col)
    add_text_box(slide, title, Inches(0.5), y + Inches(0.42), Inches(2.45), Inches(0.45),
                 font_size=Pt(13), bold=True, color=C_BG, align=PP_ALIGN.CENTER)
    add_rect(slide, Inches(3.15), y + Inches(0.15), Inches(5.5), Inches(1.02), fill_color=C_CODE_BG)
    add_text_box(slide, cmd, Inches(3.28), y + Inches(0.22), Inches(5.3), Inches(0.9),
                 font_size=Pt(13), bold=True, color=col)
    add_text_box(slide, desc, Inches(8.85), y + Inches(0.28), Inches(3.9), Inches(0.78),
                 font_size=Pt(12), color=C_GRAY)
    y += Inches(1.45)

# ══════════════════════════════════════════════
# 슬라이드 11 — 트러블슈팅 & 마무리
# ══════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
add_bg(slide)
add_rect(slide, 0, 0, Inches(0.18), H, fill_color=C_ACCENT)
add_text_box(slide, "자주 겪는 문제 & 마무리", Inches(0.4), Inches(0.25), Inches(12), Inches(0.55),
             font_size=Pt(13), bold=True, color=C_ACCENT)
add_text_box(slide, "막혔을 때 확인 리스트", Inches(0.4), Inches(0.72), Inches(12), Inches(0.7),
             font_size=Pt(32), bold=True, color=C_WHITE)

troubles = [
    ("conda: command not found",        "Miniconda 미설치 → miniconda.com 에서 설치 후 터미널 재시작",            C_ORANGE),
    ("c2pa-python 설치 실패",            "선택 라이브러리라 무시해도 됩니다. 팀장에게 문의",                        C_ACCENT2),
    ("(hnf) 가 안 붙어요",               "conda activate hnf 다시 실행. 환경 생성이 안 됐으면 STEP 3 부터 재시도", C_GREEN),
    ("7860 포트 접속 안 됨",             "python app.py 터미널이 닫혀있지 않은지 확인. 실행 중인 창 유지 필요",    C_ACCENT),
    ("git push 권한 오류",               "GitHub 토큰 만료 가능성. claude 또는 gemini 재실행 후 재로그인",         C_ORANGE),
    (".env 파일 API 오류",               "API 키 복사 시 앞뒤 공백 없는지 확인. .env 파일이 프로젝트 루트에 있는지 확인", C_ACCENT2),
]

for i, (prob, sol, col) in enumerate(troubles):
    row = i // 2
    ci  = i % 2
    x = Inches(0.4) + ci * Inches(6.5)
    y = Inches(1.58) + row * Inches(1.42)
    add_rect(slide, x, y, Inches(6.2), Inches(1.28), fill_color=C_CARD,
             line_color=col, line_width=Pt(1.5))
    add_rect(slide, x, y, Inches(0.12), Inches(1.28), fill_color=col)
    add_text_box(slide, "❓ " + prob, x + Inches(0.25), y + Inches(0.08), Inches(5.8), Inches(0.42),
                 font_size=Pt(13), bold=True, color=col)
    add_text_box(slide, "→ " + sol,   x + Inches(0.25), y + Inches(0.5), Inches(5.8), Inches(0.7),
                 font_size=Pt(11.5), color=C_GRAY)

# 마무리 메시지
add_rect(slide, Inches(0.4), Inches(6.18), Inches(12.5), Inches(1.05), fill_color=C_ACCENT)
add_text_box(slide, "환경 구성 완료 체크리스트",
             Inches(0.6), Inches(6.22), Inches(12), Inches(0.38),
             font_size=Pt(14), bold=True, color=C_BG)
add_text_box(slide,
    "☑  터미널에 (hnf) 표시    ☑  http://127.0.0.1:7860 화면 확인    ☑  .env 파일 API 키 입력    ☑  git pull 최신화",
    Inches(0.6), Inches(6.56), Inches(12.2), Inches(0.45),
    font_size=Pt(13), bold=True, color=C_BG)

out = "/Users/woochul/github/Last_Project_test/GitHub_개발환경_튜토리얼.pptx"
prs.save(out)
print("저장 완료:", out)
