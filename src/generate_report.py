import os
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

TEMPLATE_PATH = r"E:\QST\文档\400源码实验手册\混凝药剂投放量预测分析\混凝药剂投放量预测分析v1.docx"
OUTPUT_DIR = "output"


def _safe_add_heading(doc, text, level=1):
    """Add a heading, falling back to styled paragraph if style missing."""
    style_names = {0: "Title", 1: "Heading 1", 2: "Heading 2", 3: "Heading 3"}
    style_name = style_names.get(level, "Normal")
    try:
        return doc.add_heading(text, level=level)
    except KeyError:
        p = doc.add_paragraph()
        run = p.add_run(text)
        if level == 0:
            run.bold = True
            run.font.size = Pt(22)
        elif level == 1:
            run.bold = True
            run.font.size = Pt(16)
        elif level == 2:
            run.bold = True
            run.font.size = Pt(14)
        else:
            run.bold = True
            run.font.size = Pt(12)
        return p


def _safe_add_list_item(doc, text):
    try:
        return doc.add_paragraph(text, style="List Bullet")
    except KeyError:
        p = doc.add_paragraph()
        run = p.add_run("  \u2022  " + text)
        return p


def create_report(metrics, eda_images=None, prefix="experiment"):
    print("=" * 50)
    print("生成实验手册文档")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(TEMPLATE_PATH):
        doc = Document(TEMPLATE_PATH)
        print(f"[报告] 已加载模板: {TEMPLATE_PATH}")
    else:
        doc = Document()
        print(f"[报告] 模板不存在，使用默认样式")

    _safe_add_heading(doc, "心血管疾病发病风险预测分析 — KNN 模型实验报告", level=0)

    _safe_add_heading(doc, "1. 实验目的", level=1)
    doc.add_paragraph(
        "本实验旨在通过机器学习KNN（K-Nearest Neighbors）算法，"
        "基于患者特征（性别、年龄、身高、体重、血压、胆固醇、血糖、"
        "吸烟、饮酒、锻炼等）建立心血管疾病发病风险评估模型，"
        "帮助识别高风险人群并指导预防干预。同时通过实践掌握Pandas、"
        "Scikit-learn等工具在数据科学全流程中的应用。"
    )

    _safe_add_heading(doc, "2. 数据集说明", level=1)
    p = doc.add_paragraph()
    p.add_run("训练集").bold = True
    p.add_run("：历史患者数据信息.csv，共 67,849 条记录，每条记录包含患者编号、"
              "年龄、性别、身高、体重、收缩压、舒张压、胆固醇、葡萄糖、"
              "是否吸烟、是否饮酒、是否锻炼、是否有心血管疾病共13个字段。")
    p2 = doc.add_paragraph()
    p2.add_run("预测集").bold = True
    p2.add_run("：当前患者数据信息.csv，共 2,000 条记录，用于模型预测。")

    _safe_add_heading(doc, "3. 数据探索与可视化", level=1)
    doc.add_paragraph(
        "通过探索性数据分析（EDA），对数据进行了缺失值检测、"
        "异常值检测、特征分布分析、相关性分析和目标变量分布分析。"
    )

    if eda_images:
        for img_name in eda_images:
            img_path = os.path.join(OUTPUT_DIR, img_name)
            if os.path.exists(img_path):
                doc.add_picture(img_path, width=Inches(5.5))
                last_paragraph = doc.paragraphs[-1]
                last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    _safe_add_heading(doc, "4. 特征工程", level=1)
    doc.add_paragraph(
        "对分类特征进行编码转换：胆固醇和葡萄糖为三分类编码"
        "（正常=0, 高于正常=1, 远高于正常=2）；性别、吸烟、饮酒、"
        "锻炼为二分类编码（男/女, 吸烟/不吸烟等）。"
    )
    doc.add_paragraph(
        "数值特征（年龄、身高、体重、收缩压、舒张压）进行Z-score标准化。"
        "数据集按8:2比例划分为训练集和测试集。"
    )

    _safe_add_heading(doc, "5. 模型训练与调优", level=1)
    doc.add_paragraph(
        "使用KNN分类器，通过网格搜索（GridSearchCV）对以下超参数进行调优："
    )
    _safe_add_list_item(doc, "k值 (n_neighbors): 3, 5, 7, 9, 11, 13, 15, 19, 23")
    _safe_add_list_item(doc, "权重策略 (weights): uniform, distance")
    _safe_add_list_item(doc, "距离度量 (metric): euclidean, manhattan, minkowski")

    _safe_add_heading(doc, "6. 模型评估", level=1)
    if metrics:
        metric_names = {
            "accuracy": "准确率 (Accuracy)",
            "precision": "精确率 (Precision)",
            "recall": "召回率 (Recall)",
            "f1": "F1 分数",
            "auc": "AUC",
        }
        table = doc.add_table(rows=len(metric_names) + 1, cols=2)
        try:
            table.style = "Light Shading Accent 1"
        except KeyError:
            pass
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        hdr = table.rows[0].cells
        hdr[0].text = "评估指标"
        hdr[1].text = "数值"
        for i, (key, label) in enumerate(metric_names.items()):
            row = table.rows[i + 1].cells
            row[0].text = label
            row[1].text = f"{metrics.get(key, 0):.4f}"

        doc.add_paragraph("")

    _safe_add_heading(doc, "7. 预测结果", level=1)
    doc.add_paragraph(
        "使用训练好的KNN模型对当前患者数据进行预测，"
        "输出每位患者的心血管疾病发病风险（有/无）及概率值，"
        "并划分为高风险（≥70%）、中风险（40%-70%）、低风险（<40%）三个等级。"
    )

    _safe_add_heading(doc, "8. 实验总结", level=1)
    doc.add_paragraph(
        "本实验完整实现了从数据加载、探索性分析、特征工程、"
        "模型训练与调优到结果预测的全流程数据科学工作。"
        "通过KNN算法的实践，深入理解了距离度量、k值选择、"
        "特征标准化等关键概念对模型性能的影响。"
    )

    output_path = os.path.join(OUTPUT_DIR, f"{prefix}_report.docx")
    doc.save(output_path)
    print(f"[报告] 实验手册已生成: {output_path}")
    return output_path


if __name__ == "__main__":
    metrics = {"accuracy": 0.85, "precision": 0.82, "recall": 0.79, "f1": 0.80, "auc": 0.88}
    report_path = create_report(metrics)
    print(f"[报告] 生成完成: {report_path}")
