from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUTPUT_PATH = Path(r"B:\_projects\poply-image-segmentation\output\Lo_trinh_hoc_tieng_Nhat_tu_so_0_den_N4.docx")

FONT_NAME = "Times New Roman"
BODY_SIZE = 13
COLOR_H1 = "1F4E79"
COLOR_H2 = "28706A"
COLOR_H3 = "365F91"
COLOR_TABLE_HEADER = "1F4E79"
COLOR_TABLE_ALT = "EEF4F8"
COLOR_BORDER = "D9D9D9"


def set_run_font(run, size=BODY_SIZE, bold=None, italic=None, color=None):
    run.font.name = FONT_NAME
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(qn(f"w:{key}"), FONT_NAME)


def set_paragraph_body(paragraph, before=0, after=5, line_spacing=1.5):
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line_spacing
    fmt.widow_control = True


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=100, start=115, bottom=100, end=115):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color=COLOR_BORDER, size="6"):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "start" if edge == "left" else "end" if edge == "right" else edge
        element = borders.find(qn(f"w:{tag}"))
        if element is None:
            element = OxmlElement(f"w:{tag}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_cell_width(cell, width_inches):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(int(width_inches * 1440)))
    tc_w.set(qn("w:type"), "dxa")


def format_cell_text(cell, bold=False, color=None, alignment=WD_ALIGN_PARAGRAPH.LEFT, size=BODY_SIZE):
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    set_cell_margins(cell)
    for paragraph in cell.paragraphs:
        paragraph.alignment = alignment
        set_paragraph_body(paragraph, after=0, line_spacing=1.15)
        for run in paragraph.runs:
            set_run_font(run, size=size, bold=bold, color=color)


def add_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    header_row = table.rows[0]
    set_repeat_table_header(header_row)
    for idx, header in enumerate(headers):
        cell = header_row.cells[idx]
        cell.text = header
        set_cell_width(cell, widths[idx])
        set_cell_shading(cell, COLOR_TABLE_HEADER)
        format_cell_text(cell, bold=True, color="FFFFFF", alignment=WD_ALIGN_PARAGRAPH.CENTER)
    for ridx, row in enumerate(rows):
        cells = table.add_row().cells
        for cidx, value in enumerate(row):
            cells[cidx].text = value
            set_cell_width(cells[cidx], widths[cidx])
            if ridx % 2 == 1:
                set_cell_shading(cells[cidx], COLOR_TABLE_ALT)
            align = WD_ALIGN_PARAGRAPH.CENTER if cidx == 0 else WD_ALIGN_PARAGRAPH.LEFT
            format_cell_text(cells[cidx], alignment=align)
    spacer = doc.add_paragraph()
    set_paragraph_body(spacer, after=2)
    return table


def add_body(doc, text, bold_lead=None, after=5, align=WD_ALIGN_PARAGRAPH.LEFT):
    paragraph = doc.add_paragraph()
    paragraph.alignment = align
    set_paragraph_body(paragraph, after=after)
    if bold_lead and text.startswith(bold_lead):
        lead = paragraph.add_run(bold_lead)
        set_run_font(lead, bold=True)
        rest = paragraph.add_run(text[len(bold_lead):])
        set_run_font(rest)
    else:
        run = paragraph.add_run(text)
        set_run_font(run)
    return paragraph


def add_dash(doc, text, level=0, after=3):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_paragraph_body(paragraph, after=after)
    paragraph.paragraph_format.left_indent = Inches(0.28 + 0.24 * level)
    paragraph.paragraph_format.first_line_indent = Inches(-0.18)
    run = paragraph.add_run("- " + text)
    set_run_font(run)
    return paragraph


def add_labeled_dash(doc, label, text, level=0, after=3):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_paragraph_body(paragraph, after=after)
    paragraph.paragraph_format.left_indent = Inches(0.28 + 0.24 * level)
    paragraph.paragraph_format.first_line_indent = Inches(-0.18)
    dash = paragraph.add_run("- ")
    set_run_font(dash)
    lead = paragraph.add_run(label + ": ")
    set_run_font(lead, bold=True)
    body = paragraph.add_run(text)
    set_run_font(body)
    return paragraph


def add_heading(doc, text, level):
    paragraph = doc.add_paragraph(style=f"Heading {level}")
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run(text)
    if level == 1:
        set_run_font(run, size=20, bold=True, italic=False, color=COLOR_H1)
    elif level == 2:
        set_run_font(run, size=16, bold=True, italic=False, color=COLOR_H2)
    else:
        set_run_font(run, size=14, bold=True, italic=True, color=COLOR_H3)
    return paragraph


def add_hyperlink(paragraph, text, url):
    part = paragraph.part
    rel_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rel_id)
    new_run = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    rfonts = OxmlElement("w:rFonts")
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(qn(f"w:{key}"), FONT_NAME)
    size = OxmlElement("w:sz")
    size.set(qn("w:val"), str(BODY_SIZE * 2))
    r_pr.extend([rfonts, color, underline, size])
    new_run.append(r_pr)
    text_node = OxmlElement("w:t")
    text_node.text = text
    new_run.append(text_node)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)


def add_source(doc, number, title, url, description):
    paragraph = doc.add_paragraph()
    set_paragraph_body(paragraph, after=6)
    paragraph.paragraph_format.left_indent = Inches(0.3)
    paragraph.paragraph_format.first_line_indent = Inches(-0.3)
    prefix = paragraph.add_run(f"{number}. ")
    set_run_font(prefix, bold=True)
    add_hyperlink(paragraph, title, url)
    suffix = paragraph.add_run(f". {description}")
    set_run_font(suffix)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    run._r.append(fld_char1)

    instruction_run = paragraph.add_run()
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    instruction_run._r.append(instr_text)

    separate_run = paragraph.add_run()
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "separate")
    separate_run._r.append(fld_char2)

    result_run = paragraph.add_run("1")

    end_run = paragraph.add_run()
    fld_char3 = OxmlElement("w:fldChar")
    fld_char3.set(qn("w:fldCharType"), "end")
    end_run._r.append(fld_char3)

    for page_run in (run, instruction_run, separate_run, result_run, end_run):
        set_run_font(page_run, size=11, bold=True)


def configure_styles(doc):
    normal = doc.styles["Normal"]
    normal.font.name = FONT_NAME
    normal.font.size = Pt(BODY_SIZE)
    normal._element.rPr.rFonts.set(qn("w:ascii"), FONT_NAME)
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_NAME)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_NAME)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(5)

    title = doc.styles["Title"]
    title.font.name = FONT_NAME
    title.font.size = Pt(24)
    title.font.bold = True
    title.font.color.rgb = RGBColor(0, 0, 0)
    title._element.rPr.rFonts.set(qn("w:ascii"), FONT_NAME)
    title._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_NAME)
    title._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_NAME)
    title.paragraph_format.space_after = Pt(10)

    for level, size, color, italic in (
        (1, 20, COLOR_H1, False),
        (2, 16, COLOR_H2, False),
        (3, 14, COLOR_H3, True),
    ):
        style = doc.styles[f"Heading {level}"]
        style.font.name = FONT_NAME
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.italic = italic
        style.font.color.rgb = RGBColor.from_string(color)
        style._element.rPr.rFonts.set(qn("w:ascii"), FONT_NAME)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_NAME)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_NAME)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(18 if level == 1 else 13 if level == 2 else 9)
        style.paragraph_format.space_after = Pt(8 if level == 1 else 6 if level == 2 else 4)
        style.paragraph_format.line_spacing = 1.1


def add_week(doc, week_no, title, focus, materials, apps, exercises, completion):
    add_heading(doc, f"Tuần {week_no}: {title}", 3)
    add_labeled_dash(doc, "Trọng tâm", focus)
    add_labeled_dash(doc, "Tài liệu", materials)
    add_labeled_dash(doc, "App", apps)
    add_labeled_dash(doc, "Bài tập bắt buộc", exercises)
    add_labeled_dash(doc, "Hoàn thành khi", completion, after=6)


def build_document():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    configure_styles(doc)

    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)
    section.header_distance = Inches(0.3)
    section.footer_distance = Inches(0.3)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.add_run("Lộ trình học tiếng Nhật từ số 0 đến N4")
    set_run_font(title_run, size=24, bold=True, color="000000")

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_paragraph_body(subtitle, after=14, line_spacing=1.2)
    sub_run = subtitle.add_run("Chặng 1 từ người mới bắt đầu đến N5 và chặng 2 từ N5 lên N4")
    set_run_font(sub_run, size=15, bold=True, italic=True, color=COLOR_H2)

    add_body(
        doc,
        "Tài liệu này dành cho người chưa biết bảng chữ cái, chưa học ngữ pháp và chưa có vốn từ tiếng Nhật. Lộ trình kết hợp giáo trình Minna no Nihongo, tài nguyên tiếng Việt, app ôn tập, luyện nghe, luyện đọc và bài kiểm tra theo từng tuần.",
        after=7,
    )
    add_body(
        doc,
        "Kế hoạch chuẩn kéo dài 40 tuần, gồm 18 tuần đến N5 và 22 tuần đến N4. Người học nên dành 60 đến 90 phút mỗi ngày, 6 ngày mỗi tuần. Nếu chỉ học khoảng 30 đến 45 phút mỗi ngày, có thể kéo dài mỗi chặng thêm 30 đến 50 phần trăm.",
        after=10,
    )

    add_table(
        doc,
        ["Chặng học", "Thời lượng", "Giáo trình chính", "Kết quả dự kiến"],
        [
            ("Từ số 0 đến N5", "18 tuần", "Minna no Nihongo Sơ cấp I", "Đọc được kana, nắm nền ngữ pháp, nghe và đọc nội dung N5"),
            ("Từ N5 lên N4", "22 tuần", "Minna no Nihongo Sơ cấp II", "Hiểu tiếng Nhật cơ bản trong tình huống quen thuộc và làm được đề N4"),
            ("Tổng lộ trình", "40 tuần", "App và tài liệu bổ trợ", "Có nền tảng để tiếp tục học N3"),
        ],
        [1.5, 1.1, 2.0, 2.3],
    )

    add_heading(doc, "1. Cách sử dụng lộ trình", 1)
    add_heading(doc, "1.1 Điều kiện học", 2)
    add_dash(doc, "Học 6 ngày mỗi tuần và dành 1 ngày nghỉ hoặc nghe nhẹ.")
    add_dash(doc, "Mỗi buổi học phải có ít nhất hai hoạt động: học kiến thức và thực hành nghe, đọc hoặc nói.")
    add_dash(doc, "Không chuyển tuần nếu còn hơn 20 phần trăm bài tập chưa hoàn thành.")
    add_dash(doc, "Luôn ôn từ đến hạn trước khi nạp từ mới trên app.")
    add_dash(doc, "Tắt Romaji sau tuần 3. Từ đó chỉ dùng Hiragana, Katakana và Kanji có furigana khi cần.")

    add_heading(doc, "1.2 Mục tiêu đầu ra", 2)
    add_table(
        doc,
        ["Cấp độ", "Mục tiêu nghe và đọc", "Mục tiêu kiến thức", "Mục tiêu sử dụng"],
        [
            ("N5", "Hiểu hội thoại ngắn nói chậm và đoạn văn rất cơ bản", "Khoảng 700 đến 900 từ và 100 đến 150 kanji theo kế hoạch", "Tự giới thiệu, hỏi giờ, mua sắm và nói về sinh hoạt hằng ngày"),
            ("N4", "Hiểu nội dung chính của hội thoại chậm và văn bản về chủ đề quen thuộc", "Khoảng 1500 đến 2000 từ và 300 đến 350 kanji tính lũy kế", "Kể trải nghiệm, nêu kế hoạch, giải thích lý do và xử lý tình huống cơ bản"),
        ],
        [0.75, 2.05, 2.15, 2.1],
    )
    add_body(
        doc,
        "Các số lượng từ và kanji trên đây là mốc lập kế hoạch, không phải danh sách chính thức của JLPT. JLPT không công bố danh mục từ vựng, kanji và ngữ pháp cố định.",
        bold_lead="Các số lượng từ và kanji",
        after=8,
    )

    add_heading(doc, "1.3 Nguyên tắc học", 2)
    add_dash(doc, "Học kanji trong từ và câu, không học một chữ tách rời khỏi ngữ cảnh.")
    add_dash(doc, "Mỗi mẫu ngữ pháp phải có ít nhất ba câu tự đặt liên quan đến bản thân.")
    add_dash(doc, "Nghe mỗi ngày từ tuần đầu, kể cả khi chỉ hiểu một phần nhỏ.")
    add_dash(doc, "Dùng tiếng Việt để hiểu lời giải thích, nhưng không dịch từng câu khi luyện đọc.")
    add_dash(doc, "Đề thi thử dùng để tìm lỗ hổng. Không học thuộc đáp án và không luyện đề quá sớm.")

    add_heading(doc, "2. Bộ app và tài liệu", 1)
    add_heading(doc, "2.1 Bộ app chính", 2)

    add_heading(doc, "HeyJapan", 3)
    add_dash(doc, "Dùng trong 8 đến 12 tuần đầu để học kana, từ vựng, mẫu câu và hình thành thói quen.")
    add_dash(doc, "Mỗi ngày học 10 đến 20 phút, nhưng không thay thế giáo trình Minna no Nihongo.")

    add_heading(doc, "Mazii", 3)
    add_dash(doc, "Dùng xuyên suốt để tra Nhật Việt, Việt Nhật, kanji, mẫu câu và ngữ pháp.")
    add_dash(doc, "Khi tra từ, ghi lại cách đọc, nghĩa phù hợp với câu và một câu ví dụ.")

    add_heading(doc, "MochiKanji", 3)
    add_dash(doc, "Bắt đầu từ tuần 3 để ôn từ vựng và kanji bằng lặp lại ngắt quãng.")
    add_dash(doc, "Chỉ nạp 10 từ mới mỗi ngày ở N5 và 12 đến 15 từ mỗi ngày ở N4.")

    add_heading(doc, "Todaii Japanese", 3)
    add_dash(doc, "Bắt đầu bằng bài N5 từ tuần 12; chuyển sang bài N4 trong chặng N4.")
    add_dash(doc, "Nghe trước, đọc sau, tra tối đa 5 đến 8 từ quan trọng và đọc thành tiếng một lần.")

    add_heading(doc, "Migii JLPT", 3)
    add_dash(doc, "Dùng để luyện từng dạng và làm đề trong giai đoạn cuối của mỗi cấp.")
    add_dash(doc, "Không dùng điểm Migii làm chuẩn duy nhất; cần đối chiếu bằng đề luyện chính thức của JLPT.")

    add_heading(doc, "2.2 Giáo trình và tài nguyên", 2)
    add_heading(doc, "Minna no Nihongo", 3)
    add_dash(doc, "N5 dùng Sơ cấp I từ bài 1 đến bài 25; N4 dùng Sơ cấp II từ bài 26 đến bài 50.")
    add_dash(doc, "Mỗi cấp cần sách chính và Bản dịch và Giải thích ngữ pháp tiếng Việt cùng phiên bản.")
    add_dash(doc, "Nếu bắt đầu mới và mua được trọn bộ, ưu tiên bản 3. Nếu dùng bản 2, giữ toàn bộ sách cùng bản 2.")

    add_heading(doc, "Irodori tiếng Việt", 3)
    add_dash(doc, "Dùng để luyện tình huống giao tiếp thực tế, nghe và shadowing.")
    add_dash(doc, "N5 dùng Starter; N4 dùng Elementary 1 và Elementary 2 theo chủ đề phù hợp.")

    add_heading(doc, "Marugoto và Minato", 3)
    add_dash(doc, "Dùng các khóa tiếng Việt A1 và A2 để bổ sung nghe, nói và kiến thức văn hóa.")
    add_dash(doc, "Không cần hoàn thành toàn bộ nếu đã học Minna; chỉ chọn bài đúng chủ đề đang học.")

    add_heading(doc, "NHK Easy Japanese", 3)
    add_dash(doc, "Dùng audio và phần giải thích tiếng Việt từ tuần đầu.")
    add_dash(doc, "Mỗi bài nghe theo ba lượt: đoán ý, nghe cùng văn bản, sau đó nhại lại theo câu.")

    add_heading(doc, "Bài luyện chính thức JLPT", 3)
    add_dash(doc, "Dùng câu hỏi mẫu và Official Practice Workbook để kiểm tra đúng dạng và độ khó.")
    add_dash(doc, "Không tìm đề thi thật bị phát tán hoặc PDF không rõ nguồn; lời giải có thể sai và nội dung có thể vi phạm bản quyền.")

    add_heading(doc, "2.3 Cách phối hợp để không học chồng chéo", 2)
    add_dash(doc, "Minna no Nihongo là trục kiến thức và quyết định bài học trong tuần.")
    add_dash(doc, "MochiKanji chỉ chịu trách nhiệm ôn từ và kanji; không dùng thêm một app flashcard khác nếu không cần.")
    add_dash(doc, "Irodori và NHK chịu trách nhiệm nghe, nói và tình huống giao tiếp.")
    add_dash(doc, "Todaii chịu trách nhiệm đọc có ngữ cảnh từ cuối N5.")
    add_dash(doc, "Migii và đề chính thức chỉ dùng cho kiểm tra và luyện thi.")

    add_heading(doc, "3. Lịch học chuẩn", 1)
    add_heading(doc, "3.1 Một buổi học 90 phút", 2)
    add_dash(doc, "20 phút ôn từ và kanji đến hạn trên MochiKanji.")
    add_dash(doc, "25 phút học từ vựng và ngữ pháp trong Minna no Nihongo.")
    add_dash(doc, "15 phút làm bài tập trong sách hoặc app.")
    add_dash(doc, "15 phút nghe và shadowing bằng audio Minna, Irodori hoặc NHK.")
    add_dash(doc, "15 phút đọc, nói hoặc viết câu theo nội dung vừa học.")

    add_heading(doc, "3.2 Nhịp học trong tuần", 2)
    add_dash(doc, "Thứ hai đến thứ năm học bài mới và làm bài tập ngay trong ngày.")
    add_dash(doc, "Thứ sáu ôn toàn bộ từ, kanji và ngữ pháp của tuần.")
    add_dash(doc, "Thứ bảy làm bài kiểm tra ngắn, luyện nghe và hoàn thành sản phẩm của tuần.")
    add_dash(doc, "Chủ nhật nghỉ hoặc chỉ nghe nhẹ 10 đến 15 phút.")

    doc.add_page_break()
    add_heading(doc, "4. Lộ trình từ số 0 đến N5", 1)
    add_body(
        doc,
        "Chặng N5 kéo dài 18 tuần. Ba tuần đầu dành cho bảng chữ và phát âm; 12 tuần tiếp theo hoàn thành 25 bài Minna no Nihongo Sơ cấp I; ba tuần cuối củng cố kỹ năng và luyện đề.",
        after=8,
    )

    add_heading(doc, "4.1 Bảng chữ và phát âm", 2)
    add_week(
        doc, 1, "Hiragana cơ bản",
        "Năm nguyên âm và các hàng あ đến ほ; nhận biết và viết đúng nét cơ bản.",
        "HeyJapan phần Hiragana; bảng chữ trong Minna; NHK Easy Japanese phần phát âm.",
        "HeyJapan và Mazii.",
        "Viết mỗi chữ 5 lần; đọc ngẫu nhiên 46 âm; thu âm năm hàng đầu; ôn hai lượt mỗi ngày.",
        "Nhận đúng ít nhất 90 phần trăm chữ đã học mà không nhìn Romaji.",
    )
    add_week(
        doc, 2, "Hoàn thành Hiragana",
        "Các hàng còn lại; âm đục, âm bán đục, âm ghép, âm ngắt và trường âm.",
        "HeyJapan; bảng phát âm Minna; một bài NHK Easy Japanese.",
        "HeyJapan và Mazii.",
        "Chép 20 từ Hiragana; đọc thành tiếng danh sách ngẫu nhiên; phân biệt きて và きって, おばさん và おばあさん.",
        "Đọc được từ Hiragana đơn giản với độ chính xác ít nhất 90 phần trăm.",
    )
    add_week(
        doc, 3, "Katakana và bàn phím Nhật",
        "Katakana, từ vay mượn thông dụng, gõ tiếng Nhật trên điện thoại và máy tính.",
        "HeyJapan phần Katakana; danh sách đồ ăn, quốc gia và nghề nghiệp cơ bản.",
        "HeyJapan, Mazii và bắt đầu MochiKanji.",
        "Viết tên mình bằng Katakana; gõ 30 từ; đọc nhãn sản phẩm đơn giản; ôn 10 từ mới mỗi ngày.",
        "Đọc được Katakana cơ bản và không còn dùng Romaji khi học bài mới.",
    )

    add_heading(doc, "4.2 Nền ngữ pháp N5", 2)
    add_week(
        doc, 4, "Minna bài 1 và 2",
        "Câu danh từ với です; phủ định; câu hỏi か; trợ từ は, も, の; chỉ định これ, それ, あれ.",
        "Minna I sách chính và giải thích tiếng Việt; workbook bài 1 và 2.",
        "HeyJapan, Mazii và MochiKanji.",
        "Làm bài tập A và B; viết 10 câu; thu âm phần giới thiệu bản thân 45 đến 60 giây.",
        "Tự giới thiệu tên, nghề nghiệp, quốc tịch và hỏi thông tin tương ứng.",
    )
    add_week(
        doc, 5, "Minna bài 3 và 4",
        "Địa điểm; số đếm; giờ; ngày; động từ hiện tại, phủ định, quá khứ và trợ từ thời gian に.",
        "Minna I bài 3 và 4; workbook; audio hội thoại.",
        "HeyJapan, Mazii và MochiKanji.",
        "Lập thời gian biểu một ngày; hỏi đáp giờ mở cửa; nghe audio rồi chép các mốc thời gian.",
        "Nói được lịch sinh hoạt và nhận đúng giờ, ngày trong đoạn nghe chậm.",
    )
    add_week(
        doc, 6, "Minna bài 5 và 6",
        "Di chuyển với へ, に, で; tân ngữ を; phương tiện, người đi cùng và hoạt động hằng ngày.",
        "Minna I bài 5 và 6; Irodori Starter chủ đề đi lại và sinh hoạt.",
        "Mazii, MochiKanji và Irodori.",
        "Viết lịch chuyến đi; nói 10 câu về một ngày; shadowing hai đoạn hội thoại.",
        "Phân biệt được nơi đến, phương tiện, địa điểm thực hiện hành động và tân ngữ.",
    )
    add_week(
        doc, 7, "Minna bài 7 và 8",
        "Phương tiện thực hiện hành động với で; cho nhận đồ vật; tính từ đuôi い và な.",
        "Minna I bài 7 và 8; workbook; audio.",
        "Mazii, MochiKanji và HeyJapan.",
        "Mô tả 10 đồ vật; viết đoạn 6 câu về thành phố hoặc căn phòng; làm bài phân biệt tính từ.",
        "Dùng đúng tính từ ở hiện tại khẳng định và phủ định trong câu đơn.",
    )
    add_week(
        doc, 8, "Minna bài 9 và 10",
        "Sở thích và năng lực với が; lý do から; tồn tại あります, います; vị trí đồ vật và người.",
        "Minna I bài 9 và 10; Irodori Starter chủ đề sở thích và nơi ở.",
        "Mazii, MochiKanji và Irodori.",
        "Vẽ sơ đồ phòng và mô tả; nói về sở thích; làm bài kiểm tra tổng hợp bài 1 đến 10.",
        "Bài kiểm tra tổng hợp đạt ít nhất 75 phần trăm.",
    )
    add_week(
        doc, 9, "Minna bài 11 và 12",
        "Lượng từ; tần suất; khoảng thời gian; so sánh; tính từ và danh từ ở quá khứ.",
        "Minna I bài 11 và 12; workbook; audio có số lượng và thời gian.",
        "Mazii và MochiKanji.",
        "Luyện 20 câu lượng từ; so sánh ba địa điểm; nghe và ghi số lượng, giá, thời gian.",
        "Trả lời chính xác các câu hỏi bao nhiêu, bao lâu và cái nào hơn.",
    )
    add_week(
        doc, 10, "Minna bài 13 và 14",
        "Mong muốn ほしい và たい; mục đích di chuyển; thể て; yêu cầu với てください.",
        "Minna I bài 13 và 14; bảng chia thể て; workbook.",
        "Mazii, MochiKanji và HeyJapan.",
        "Chia 30 động từ sang thể て; viết kế hoạch cuối tuần; đóng vai gọi món hoặc nhờ giúp đỡ.",
        "Chia thể て đúng ít nhất 85 phần trăm và dùng được trong yêu cầu đơn giản.",
    )
    add_week(
        doc, 11, "Minna bài 15 và 16",
        "Cho phép, cấm đoán, trạng thái với ています; nối hành động bằng thể て và てから.",
        "Minna I bài 15 và 16; Irodori chủ đề nội quy và sinh hoạt.",
        "Mazii, MochiKanji và Irodori.",
        "Viết nội quy 8 câu; kể trình tự buổi sáng; shadowing hai hội thoại.",
        "Mô tả được hành động đang diễn ra, thói quen và chuỗi hành động.",
    )
    add_week(
        doc, 12, "Minna bài 17 và 18",
        "Thể ない; nghĩa vụ và không cần thiết; thể từ điển; khả năng với ことができます; sở thích.",
        "Minna I bài 17 và 18; workbook; Todaii bài đọc N5 rất ngắn.",
        "Mazii, MochiKanji và bắt đầu Todaii.",
        "Chia 30 động từ sang thể ない và từ điển; viết lời khuyên sức khỏe; đọc hai bài N5.",
        "Phân biệt được điều phải làm, không được làm và có thể làm.",
    )
    add_week(
        doc, 13, "Minna bài 19 và 20",
        "Thể た; kinh nghiệm; liệt kê hành động; thay đổi trạng thái; thể thông thường.",
        "Minna I bài 19 và 20; bảng đối chiếu thể lịch sự và thể thông thường.",
        "Mazii, MochiKanji và Todaii.",
        "Viết 10 câu về trải nghiệm; chuyển 20 câu sang thể thông thường; thu âm hội thoại với bạn bè.",
        "Dùng được thể thông thường cơ bản mà không trộn sai với です và ます.",
    )
    add_week(
        doc, 14, "Minna bài 21 và 22",
        "Nêu ý kiến với と思います; thuật lại lời nói với と言います; mệnh đề bổ nghĩa danh từ.",
        "Minna I bài 21 và 22; workbook; hai bài đọc N5.",
        "Mazii, MochiKanji và Todaii.",
        "Viết đoạn nêu ý kiến 8 câu; ghép 15 cặp câu thành mệnh đề bổ nghĩa; đọc thành tiếng.",
        "Hiểu ai làm gì trong câu có mệnh đề bổ nghĩa ngắn.",
    )
    add_week(
        doc, 15, "Minna bài 23 đến 25",
        "Mốc thời gian とき; điều kiện と và たら; nhượng bộ ても; cho nhận hành động.",
        "Minna I bài 23 đến 25; workbook; audio tổng hợp.",
        "Mazii, MochiKanji và Irodori.",
        "Lập hướng dẫn sử dụng một thiết bị; viết tình huống nếu; kể một lần được người khác giúp.",
        "Hoàn thành toàn bộ bài 1 đến 25 và dùng được các mẫu câu trọng tâm trong câu tự đặt.",
    )

    add_heading(doc, "4.3 Củng cố và luyện thi N5", 2)
    add_week(
        doc, 16, "Hệ thống hóa N5",
        "Trợ từ, bốn nhóm chia động từ đã học, tính từ, lượng từ và mẫu câu dễ nhầm.",
        "Ghi chú 25 bài Minna; workbook; danh sách lỗi cá nhân.",
        "Mazii và MochiKanji.",
        "Lập một trang tổng hợp trợ từ; chia 50 động từ; làm lại toàn bộ câu sai; viết đoạn 120 đến 150 chữ Nhật.",
        "Các bài kiểm tra ngữ pháp và từ vựng đạt ít nhất 80 phần trăm.",
    )
    add_week(
        doc, 17, "Đọc và nghe theo dạng N5",
        "Đọc đoạn ngắn, tìm thông tin; nghe hiểu nhiệm vụ, ý chính, phản hồi nhanh.",
        "JLPT sample questions; Todaii N5; audio Minna và NHK.",
        "Todaii và Migii JLPT.",
        "Làm từng phần có bấm giờ; sau mỗi bài ghi lý do sai; nghe lại và chép câu không hiểu.",
        "Hoàn thành trong thời gian và từng phần đạt khoảng 65 đến 70 phần trăm.",
    )
    add_week(
        doc, 18, "Thi thử và sửa lỗ hổng",
        "Mô phỏng đầy đủ N5, phân tích điểm và ôn đúng phần yếu.",
        "Official Practice Workbook N5 hoặc câu hỏi mẫu chính thức.",
        "Migii JLPT chỉ dùng bổ sung.",
        "Làm một đề không tra từ; dành hai buổi sửa đề; làm lại toàn bộ câu sai sau ba ngày.",
        "Đạt ít nhất 100 trên 180 theo mục tiêu an toàn và không có phần nào quá yếu.",
    )

    doc.add_page_break()
    add_heading(doc, "5. Lộ trình từ N5 lên N4", 1)
    add_body(
        doc,
        "Chặng N4 kéo dài 22 tuần và bắt đầu sau khi người học đã hoàn thành 25 bài Minna no Nihongo Sơ cấp I. Trọng tâm chuyển từ câu đơn sang câu ghép, mô tả tình huống, đọc đoạn dài hơn và nghe nội dung hằng ngày.",
        after=8,
    )

    add_heading(doc, "5.1 Hoàn thành Minna no Nihongo Sơ cấp II", 2)
    add_week(
        doc, 1, "Ôn N5 và Minna bài 26",
        "Kiểm tra lại trợ từ và chia động từ; mẫu giải thích んです; yêu cầu lời khuyên.",
        "Minna II bài 26; ghi chú N5; một đề chẩn đoán N4 ngắn.",
        "Mazii, MochiKanji và Todaii.",
        "Làm đề chẩn đoán; lập danh sách ba nhóm điểm yếu; viết 10 câu với んです.",
        "Xác định rõ phần N5 cần ôn song song và hoàn thành bài 26.",
    )
    add_week(
        doc, 2, "Minna bài 27 và 28",
        "Thể khả năng; nhìn thấy và nghe thấy; ながら; thói quen; liệt kê lý do với し.",
        "Minna II bài 27 và 28; workbook; Irodori A2 chủ đề khả năng và công việc.",
        "Mazii, MochiKanji và Irodori.",
        "Viết đoạn mô tả khả năng; kể lịch làm việc có hai hoạt động đồng thời; shadowing.",
        "Dùng đúng thể khả năng và phân biệt 見えます với 見られます ở mức cơ bản.",
    )
    add_week(
        doc, 3, "Minna bài 29 và 30",
        "Trạng thái kết quả với ています; chuẩn bị và sắp đặt với てあります và ておきます.",
        "Minna II bài 29 và 30; workbook; bài đọc về chuẩn bị sự kiện.",
        "Mazii, MochiKanji và Todaii.",
        "Mô tả 10 trạng thái trong phòng; lập danh sách chuẩn bị cho chuyến đi; đọc hai bài N4.",
        "Phân biệt hành động đang diễn ra, trạng thái còn lại và hành động chuẩn bị.",
    )
    add_week(
        doc, 4, "Minna bài 31 và 32",
        "Ý chí, dự định và kế hoạch; lời khuyên; dự đoán với でしょう và かもしれません.",
        "Minna II bài 31 và 32; Irodori chủ đề kế hoạch và sức khỏe.",
        "Mazii, MochiKanji và Irodori.",
        "Trình bày kế hoạch ba tháng; viết lời khuyên sức khỏe; nghe và xác định mức chắc chắn.",
        "Nêu được ý định, lịch đã định và dự đoán với mức độ phù hợp.",
    )
    add_week(
        doc, 5, "Minna bài 33 và 34",
        "Mệnh lệnh, cấm đoán và ý nghĩa ký hiệu; làm theo hướng dẫn với とおり; trình tự với あとで.",
        "Minna II bài 33 và 34; workbook; biển báo và hướng dẫn thực tế.",
        "Mazii, MochiKanji và Todaii.",
        "Giải thích 10 biển báo; viết hướng dẫn 8 bước; luyện nghe chỉ dẫn.",
        "Hiểu được chỉ dẫn ngắn, biển báo và trình tự thực hiện.",
    )
    add_week(
        doc, 6, "Ôn bài 26 đến 34",
        "Củng cố thể khả năng, trạng thái, chuẩn bị, ý định, dự đoán và hướng dẫn.",
        "Workbook và danh sách lỗi bài 26 đến 34.",
        "MochiKanji, Todaii và Migii phần bài tập ngắn.",
        "Làm bài kiểm tra tổng hợp; viết đoạn 180 chữ Nhật; thu âm bài nói hai phút.",
        "Bài kiểm tra đạt ít nhất 75 phần trăm và giải thích được lỗi sai chính.",
    )
    add_week(
        doc, 7, "Minna bài 35 và 36",
        "Điều kiện ば; các mẫu ように, ようになります và ようにします.",
        "Minna II bài 35 và 36; workbook; Irodori A2 chủ đề thay đổi thói quen.",
        "Mazii, MochiKanji và Irodori.",
        "Chia 30 từ sang thể ば; viết kế hoạch thay đổi thói quen; nói về điều đã có thể làm.",
        "Phân biệt điều kiện và mẫu diễn tả thay đổi khả năng, thói quen.",
    )
    add_week(
        doc, 8, "Minna bài 37 và 38",
        "Thể bị động; danh từ hóa với の; diễn đạt thích, ghét và quên một hành động.",
        "Minna II bài 37 và 38; workbook; bài đọc N4.",
        "Mazii, MochiKanji và Todaii.",
        "Chuyển 20 câu sang bị động; viết đoạn về một công trình; làm bài phân biệt の và こと.",
        "Nhận diện chủ thể và đối tượng đúng trong câu bị động cơ bản.",
    )
    add_week(
        doc, 9, "Minna bài 39 và 40",
        "Nguyên nhân và cảm xúc; câu hỏi lồng với か và かどうか; thử thực hiện với てみます.",
        "Minna II bài 39 và 40; workbook; audio hội thoại.",
        "Mazii, MochiKanji và Todaii.",
        "Viết giải thích nguyên nhân; chuyển 15 câu hỏi thành câu hỏi lồng; nghe và ghi thông tin cần xác nhận.",
        "Dùng được câu hỏi lồng và nêu nguyên nhân mà không nhầm trật tự câu.",
    )
    add_week(
        doc, 10, "Minna bài 41 và 42",
        "Cho nhận ở mức lịch sự; mục đích, công dụng, vật liệu, chi phí và thời gian cần thiết.",
        "Minna II bài 41 và 42; Irodori chủ đề nhờ giúp đỡ và đồ dùng.",
        "Mazii, MochiKanji và Irodori.",
        "Viết thư cảm ơn ngắn; mô tả công dụng năm đồ vật; hỏi đáp chi phí và thời gian.",
        "Chọn đúng mẫu cho nhận theo quan hệ người nói và người nghe.",
    )
    add_week(
        doc, 11, "Minna bài 43 và 44",
        "Dấu hiệu với そうです; quá mức với すぎます; dễ và khó; điều chỉnh trạng thái.",
        "Minna II bài 43 và 44; workbook; bài đọc mô tả sản phẩm và thời tiết.",
        "Mazii, MochiKanji và Todaii.",
        "Mô tả 10 hình ảnh; viết đánh giá sản phẩm; luyện phân biệt そうです với các mẫu đã học.",
        "Nói được cảm nhận dựa trên dấu hiệu và yêu cầu điều chỉnh đơn giản.",
    )
    add_week(
        doc, 12, "Ôn bài 35 đến 44",
        "Hệ thống điều kiện, thay đổi, bị động, câu hỏi lồng, cho nhận và biểu hiện trạng thái.",
        "Workbook; ghi chú bài 35 đến 44; bài đọc N4.",
        "MochiKanji, Todaii và Migii bài tập ngắn.",
        "Làm bài tổng hợp; sửa toàn bộ câu sai; viết đoạn 200 chữ Nhật sử dụng ít nhất sáu mẫu.",
        "Bài tổng hợp đạt ít nhất 75 phần trăm.",
    )
    add_week(
        doc, 13, "Minna bài 45 và 46",
        "Trường hợp với 場合; nhượng bộ のに; thời điểm hành động với ところ; vừa mới với ばかり; phán đoán はず.",
        "Minna II bài 45 và 46; workbook; email và thông báo N4.",
        "Mazii, MochiKanji và Todaii.",
        "Viết hướng dẫn xử lý ba tình huống; phân biệt ところ, ばかり và はず qua 20 câu.",
        "Hiểu được tiến trình thời gian và phán đoán trong câu.",
    )
    add_week(
        doc, 14, "Minna bài 47 và 48",
        "Nghe nói và truyền đạt với そうです; phán đoán ようです; thể sai khiến và nhờ cho phép.",
        "Minna II bài 47 và 48; workbook; audio hội thoại.",
        "Mazii, MochiKanji và Irodori.",
        "Tóm tắt một tin ngắn; chuyển 20 động từ sang sai khiến; đóng vai xin phép trong công việc.",
        "Phân biệt thông tin nghe lại, phán đoán và yêu cầu sai khiến cơ bản.",
    )
    add_week(
        doc, 15, "Minna bài 49",
        "Tôn kính ngữ, cách nói về hành động của khách hàng, cấp trên và người cần tôn trọng.",
        "Minna II bài 49; bảng tôn kính ngữ; hội thoại dịch vụ.",
        "Mazii và Irodori.",
        "Chuyển 20 câu sang tôn kính ngữ; đóng vai tiếp khách; nghe và đánh dấu từ tôn kính.",
        "Nhận biết và dùng được các mẫu tôn kính cơ bản trong tình huống rõ ràng.",
    )
    add_week(
        doc, 16, "Minna bài 50",
        "Khiêm nhường ngữ, cách nói về hành động của bản thân và người trong nhóm mình.",
        "Minna II bài 50; bảng khiêm nhường ngữ; hội thoại công việc cơ bản.",
        "Mazii và Irodori.",
        "Viết email tự giới thiệu lịch sự; chuyển 20 câu; thu âm hội thoại tôn kính và khiêm nhường.",
        "Phân biệt được đối tượng cần nâng lên và hành động cần hạ mình.",
    )

    add_heading(doc, "5.2 Củng cố kỹ năng N4", 2)
    add_week(
        doc, 17, "Hệ thống ngữ pháp và trợ từ",
        "Ôn bài 26 đến 50; so sánh các mẫu gần nghĩa; sửa lỗi trợ từ và chia thể.",
        "Ghi chú Minna II; workbook; danh sách lỗi cá nhân.",
        "Mazii và Migii phần ngữ pháp.",
        "Tạo bảng 30 mẫu trọng tâm; làm lại câu sai; tự giải thích bằng tiếng Việt rồi đặt câu tiếng Nhật.",
        "Đạt ít nhất 80 phần trăm ở bài kiểm tra ngữ pháp tổng hợp.",
    )
    add_week(
        doc, 18, "Từ vựng và kanji N4",
        "Ôn từ theo chủ đề và kanji trong từ; tập phân biệt cách đọc theo ngữ cảnh.",
        "Sổ từ cá nhân; danh sách từ trong Minna II; bài luyện kanji N4.",
        "MochiKanji và Mazii.",
        "Ôn toàn bộ từ đến hạn; viết 20 câu dùng từ hay nhầm; làm hai bài kiểm tra từ và kanji.",
        "Hai bài kiểm tra liên tiếp đạt ít nhất 80 phần trăm.",
    )
    add_week(
        doc, 19, "Đọc hiểu N4",
        "Xác định chủ đề, đại từ chỉ định, từ nối, ý chính và thông tin cần tìm.",
        "Todaii N4; bài đọc Minna; JLPT sample questions N4.",
        "Todaii, Mazii và Migii.",
        "Mỗi ngày một bài đọc; gạch chân từ nối; tóm tắt một câu; hai buổi đọc có bấm giờ.",
        "Đạt khoảng 70 phần trăm mà không dịch toàn bộ sang tiếng Việt.",
    )
    add_week(
        doc, 20, "Nghe hiểu N4",
        "Nắm nhiệm vụ, ý chính, quan hệ người nói, từ phủ định và phản hồi nhanh.",
        "Audio Minna II; Irodori; NHK; JLPT sample questions N4.",
        "Todaii và Migii.",
        "Nghe một lần làm bài; lần hai chép từ khóa; lần ba nhìn script và shadowing.",
        "Ba bài nghe liên tiếp đạt ít nhất 65 đến 70 phần trăm.",
    )
    add_week(
        doc, 21, "Luyện từng phần có bấm giờ",
        "Làm quen thời gian của phần từ vựng, ngữ pháp đọc và nghe; ưu tiên phần yếu.",
        "Official Practice Workbook N4; bộ câu hỏi mẫu JLPT.",
        "Migii JLPT dùng bổ sung và lưu lịch sử lỗi.",
        "Làm từng phần riêng; ghi loại lỗi; ôn lại sau 48 giờ; làm lại câu sai mà không nhìn đáp án.",
        "Hoàn thành đúng thời gian và không phần nào dưới khoảng 60 phần trăm.",
    )
    add_week(
        doc, 22, "Thi thử N4 và kế hoạch sau N4",
        "Mô phỏng đầy đủ bài thi; xác nhận nền tảng trước khi chuyển sang N3.",
        "Official Practice Workbook N4 hoặc bộ câu hỏi chính thức.",
        "Migii JLPT chỉ dùng thêm một đề để so sánh.",
        "Làm hai đề cách nhau ít nhất ba ngày; sửa chi tiết; lập danh sách kiến thức cần ôn trong tháng kế tiếp.",
        "Hai đề đạt ít nhất 110 trên 180 theo mục tiêu an toàn và không bỏ phần thi nào.",
    )

    add_heading(doc, "6. Bài tập bắt buộc xuyên suốt", 1)
    add_heading(doc, "6.1 Sổ lỗi cá nhân", 2)
    add_dash(doc, "Mỗi lỗi ghi câu hỏi, đáp án đã chọn, đáp án đúng và lý do sai.")
    add_dash(doc, "Phân loại lỗi thành từ vựng, kanji, trợ từ, chia động từ, ngữ pháp, đọc hoặc nghe.")
    add_dash(doc, "Làm lại câu sai sau 48 giờ và sau một tuần.")

    add_heading(doc, "6.2 Sản phẩm mỗi tuần", 2)
    add_dash(doc, "Một đoạn viết từ 5 đến 10 câu theo trình độ hiện tại.")
    add_dash(doc, "Một bản ghi âm từ 1 đến 3 phút; nghe lại và ghi ba lỗi phát âm hoặc ngữ pháp.")
    add_dash(doc, "Một bài nghe được shadowing ít nhất ba lần.")
    add_dash(doc, "Một bài kiểm tra ngắn và bản sửa lỗi đầy đủ.")

    add_heading(doc, "6.3 Quy tắc tăng hoặc giảm khối lượng", 2)
    add_dash(doc, "Nếu tỷ lệ đúng dưới 70 phần trăm, giữ nguyên lượng từ mới và tăng một buổi ôn.")
    add_dash(doc, "Nếu bỏ ôn ba ngày liên tiếp, dừng nạp từ mới cho đến khi xử lý xong phần đến hạn.")
    add_dash(doc, "Nếu bài tập liên tục trên 90 phần trăm và hoàn thành sớm, tăng đọc hoặc nghe thay vì tăng mạnh số từ mới.")

    add_heading(doc, "7. Tiêu chí hoàn thành", 1)
    add_heading(doc, "7.1 Hoàn thành N5", 2)
    add_dash(doc, "Đọc và gõ được Hiragana, Katakana mà không dựa vào Romaji.")
    add_dash(doc, "Hoàn thành Minna no Nihongo Sơ cấp I và workbook đã chọn.")
    add_dash(doc, "Tự giới thiệu, nói về lịch trình, sở thích, gia đình và nhu cầu cơ bản.")
    add_dash(doc, "Làm đề N5 mục tiêu ít nhất 100 trên 180 và không có phần quá yếu.")

    add_heading(doc, "7.2 Hoàn thành N4", 2)
    add_dash(doc, "Hoàn thành Minna no Nihongo Sơ cấp II và sửa lại toàn bộ bài kiểm tra sai.")
    add_dash(doc, "Đọc được email, thông báo và đoạn văn quen thuộc mà không dịch từng câu.")
    add_dash(doc, "Nghe được nội dung chính của hội thoại nói chậm trong đời sống hằng ngày.")
    add_dash(doc, "Làm hai đề N4 mục tiêu ít nhất 110 trên 180 và hoàn thành đúng thời gian.")

    doc.add_page_break()
    add_heading(doc, "8. Nguồn và liên kết chính thức", 1)
    add_body(doc, "Các liên kết dưới đây dùng để tải tài liệu, kiểm tra tính năng app và đối chiếu cấu trúc JLPT.", after=8)
    sources = [
        ("JLPT Summary of Linguistic Competence", "https://www.jlpt.jp/e/about/levelsummary.html", "Mô tả năng lực N5 và N4."),
        ("JLPT Composition of Test Sections and Items", "https://www.jlpt.jp/e/guideline/testsections.html", "Cấu trúc và thời gian các phần thi."),
        ("JLPT Scoring Sections and Pass Marks", "https://www.jlpt.jp/e/guideline/results.html", "Điểm đỗ tổng và điểm tối thiểu từng phần."),
        ("JLPT Frequently Asked Questions", "https://www.jlpt.jp/e/faq/", "Giải thích về danh sách kiến thức và Official Practice Workbook."),
        ("Minna no Nihongo Third Edition", "https://www.3anet.co.jp/en/minnanonihongo_dai3pan.html", "Thông tin phiên bản, sách giải thích tiếng Việt và mức tham chiếu N5 N4."),
        ("Irodori tiếng Việt", "https://hn.jpf.go.jp/irodori", "Giáo trình PDF và audio tiếng Nhật trong đời sống."),
        ("Marugoto e Learning", "https://marugoto.jpf.go.jp/en/e-learning/", "Khóa tự học có hỗ trợ tiếng Việt."),
        ("NHK Easy Japanese", "https://www.jpf.go.jp/j/urawa/j_rsorcs/easyjapanese.html", "Bài nghe và hội thoại cho người mới."),
        ("HeyJapan", "https://play.google.com/store/apps/details?id=com.eup.heyjapan", "App học tiếng Nhật cho người mới."),
        ("Mazii", "https://mazii.net/vi-VN", "Từ điển Nhật Việt và công cụ học tập."),
        ("MochiKanji", "https://play.google.com/store/apps/details?id=com.mochimochi.android.an", "Ôn từ vựng và kanji."),
        ("Todaii Japanese", "https://japanese.todaiinews.com/?hl=vi-VN", "Luyện đọc và nghe theo trình độ."),
        ("Migii JLPT", "https://jlpt.migii.net/", "Bài tập và đề thi thử JLPT."),
    ]
    for idx, (name, url, description) in enumerate(sources, start=1):
        add_source(doc, idx, name, url, description)

    for sec in doc.sections:
        footer = sec.footer
        footer_para = footer.paragraphs[0]
        add_page_number(footer_para)

    core = doc.core_properties
    core.title = "Lộ trình học tiếng Nhật từ số 0 đến N4"
    core.subject = "Lộ trình N5 và N4 cho người mới bắt đầu"
    core.keywords = "tiếng Nhật, N5, N4, JLPT, lộ trình học, tiếng Việt"
    core.author = ""
    core.last_modified_by = ""

    doc.save(OUTPUT_PATH)
    print(str(OUTPUT_PATH))


if __name__ == "__main__":
    build_document()
