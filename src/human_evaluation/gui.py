from tkinter import messagebox
import tkinter as tk
import csv
import random

# 创建主窗口
root = tk.Tk()
root.title("RLCF Annotation")

# 创建文本框和标签
text_labels = []
text_boxes = []

# 创建12个文本框按照3x4的形式排列
for row in range(1):
    for col in range(4):
        index = row * 4 + col
        text_label = tk.Label(root, text=f"Document {index+1}:")
        text_label.grid(row=row, column=col * 2, padx=10, pady=5)
        text_labels.append(text_label)
        
        text_box = tk.Text(root, wrap=tk.WORD, height=10, width=20, state=tk.DISABLED)
        text_box.grid(row=row, column=col * 2 + 1, padx=10, pady=5)
        text_boxes.append(text_box)

for row in range(1,3):
    if row == 1:
        model = "A"
    else:
        model = "B"
    for col in range(4):
        if col==0:
            text_label = tk.Label(root, text=f"Model {model}: Summary {col+1}:")
        else:
            text_label = tk.Label(root, text=f"Summary {col+1}:")
        text_label.grid(row=row, column=col * 2, padx=10, pady=5)
        text_labels.append(text_label)
        
        text_box = tk.Text(root, wrap=tk.WORD, height=10, width=20, state=tk.DISABLED)
        text_box.grid(row=row, column=col * 2 + 1, padx=10, pady=5)
        text_boxes.append(text_box)

rule_text = '''Annotation Guideline

The final decision must be made after comprehensive evaluation on two group of summaries.

Specificity. The concept of specificity refers to the ability of the summary to distinguish itself from similar documents. It requires the summary to highlight unique and critical points that set it apart from other similar documents.

Correctness. Correctness, in the context of a summary, entails the accuracy and completeness of the information presented. A correct summary faithfully represents the original content without distorting the meaning, omitting vital details, or introducing any inaccuracies. It should also cover all the essential points and arguments made in the original document, hence ensuring completeness.

Concision. Concision concerns the brevity and succinctness of the summary. A concise summary effectively conveys the main points and arguments of the original document in as few words as possible, without losing critical information or context. It requires careful word choice and sentence construction to eliminate redundancy and verbosity.
'''
rule_box = tk.Text(root, wrap=tk.WORD, height=30, width=20)
rule_box.grid(row=1, column=8, columnspan=4, padx=10, pady=5)
rule_box.insert(tk.END, rule_text)

# 从CSV文件中读取数据并将其填充到文本框中
data = []  # 存储CSV数据
current_index = 0  # 当前比较的数据索引

def load_data():
    with open("data_20000.csv", "r", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        data.extend(list(csv_reader))

FLAG = 0
def show_data(index):
    for i in range(4):
        text_boxes[i].config(state=tk.NORMAL)  # 设置为可编辑
        text_boxes[i].delete("1.0", tk.END) # 清空文本框
        text_boxes[i].insert(tk.END, data[index][i])  # 填充文本框
        text_boxes[i].config(state=tk.DISABLED)  # 设置为不可编辑
    global FLAG
    FLAG = random.randint(0,1)
    for i in range(4,8):
        text_boxes[i].config(state=tk.NORMAL)  # 设置为可编辑
        text_boxes[i].delete("1.0", tk.END) # 清空文本框
        text_boxes[i].insert(tk.END, data[index][i+FLAG*4])  # 填充文本框
        text_boxes[i].config(state=tk.DISABLED)  # 设置为不可编辑
    for i in range(8,12):
        text_boxes[i].config(state=tk.NORMAL)  # 设置为可编辑
        text_boxes[i].delete("1.0", tk.END) # 清空文本框
        text_boxes[i].insert(tk.END, data[index][i-FLAG*4])  # 填充文本框
        text_boxes[i].config(state=tk.DISABLED)  # 设置为不可编辑

load_data()
random_index = random.randint(0, len(data) - 1)
previous_indexes = [random_index]
show_data(random_index)

results=[]
# 创建按钮
def next_data(result):
    global results, FLAG
    global current_index, random_index
    previous_indexes.append(random_index)
    random_index = random.randint(0, len(data) - 1)
    while random_index == previous_indexes[-1]:
        random_index = random.randint(0, len(data) - 1)
    if FLAG == 1:
        result = -result
    results.append(result)
    current_index += 1
    if current_index < len(data):
        show_data(random_index)
        update_progress()
    else:
        tk.messagebox.showinfo("提示", "标注完成，可以保存结果。")
        save_results()  # 保存结果到results.csv
        root.quit()  # 结束程序

def save_results():
    total_count = max(len(results), 1)
    model_a_better_count = sum(1 for item in results if item == 1)
    model_b_better_count = sum(1 for item in results if item == -1)
    equal_count = sum(1 for item in results if item == 0)
    
    # 计算比例
    model_a_better_ratio = model_a_better_count / total_count * 100
    model_b_better_ratio = model_b_better_count / total_count * 100
    equal_ratio = equal_count / total_count * 100
    
    # 弹出消息框显示比例
    message = f"Vanilla模型好的比例: {model_a_better_ratio:.2f}%\n"
    message += f"RLCF-optimized模型好的比例: {model_b_better_ratio:.2f}%\n"
    message += f"一样好的比例: {equal_ratio:.2f}%"
    
    messagebox.showinfo("结果比例", message)

    with open("results.csv", "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        for item in results:
            csv_writer.writerow(str(item))

def previous_data():
    global current_index, results, random_index
    if current_index > 0:
        results.pop()
        current_index -= 1
        show_data(previous_indexes.pop())
        update_progress()

def skip_data():
    global random_index
    if len(data) > 0:
        random_index = random.randint(0, len(data) - 1)
        show_data(random_index)
        update_progress()

def end_program():
    if tk.messagebox.askyesno("确认", "您确定要结束程序吗？"):
        save_results()  # 保存结果到results.csv
        root.quit()  # 结束程序

# 创建标注进度标签
progress_label = tk.Label(root, text="Process: 0/0")
progress_label.grid(row=3, column=7, padx=10, pady=5, sticky="ne")

def update_progress():
    progress_label.config(text=f"Process: {current_index + 1}/{len(data)}")

update_progress()
button1 = tk.Button(root, text="LLM A is better", command=lambda:next_data(1))
button1.grid(row=3, column=0, padx=10, pady=10)

button2 = tk.Button(root, text="The quality is the same", command=lambda:next_data(0))
button2.grid(row=3, column=1, padx=10, pady=10)

button3 = tk.Button(root, text="LLM B is better", command=lambda:next_data(-1))
button3.grid(row=3, column=2, padx=10, pady=10)

button4 = tk.Button(root, text="Return to the last one", command=previous_data)
button4.grid(row=3, column=3, padx=10, pady=10)

skip_button = tk.Button(root, text="Skip current sample", command=skip_data)
skip_button.grid(row=3, column=4, padx=10, pady=10)

save_button = tk.Button(root, text="Save", command=save_results)
save_button.grid(row=0, column=8, padx=10, pady=5, sticky="ne")
save_button.config(bg="blue", fg="blue")  # 修改按钮颜色


end_button = tk.Button(root, text="End", command=end_program)
end_button.grid(row=0, column=9, padx=10, pady=5, sticky="ne")
end_button.config(bg="red", fg="red")  # 修改按钮颜色

# 启动主循环
root.mainloop()
