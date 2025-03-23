import pandas as pd


def compare_excel_csv(excel_file, csv_file):
    # 1. 读取 Excel 文件（假设列名为 'x' 和 'y'，但实际中这两列数据是颠倒的）
    df_excel = pd.read_excel(excel_file)
    print("Excel数据：")
    print(df_excel)

    # 因为 Excel 中的数据顺序颠倒，所以这里交换顺序：原 'y' 列实际上是 x，原 'x' 列实际上是 y
    excel_set = set(df_excel.apply(lambda row: (row['y'], row['x']), axis=1))

    # 2. 读取 CSV 文件（假设第一列为索引，后两列为 'X' 和 'Y'）
    df_csv = pd.read_csv(csv_file)
    print("CSV数据：")
    print(df_csv)
    if 'Index' in df_csv.columns:
        df_csv = df_csv[['X', 'Y']]  # 只保留 X, Y 列

    # 3. 遍历 CSV 的每一行，与 Excel 数据（集合）进行比对
    matched_count = 0
    matched_rows = []

    for idx, row in df_csv.iterrows():
        # 直接比对 CSV 中的 (X, Y) 是否存在于 Excel（调换后）的集合中
        if (row['X'], row['Y']) in excel_set:
            matched_count += 1
            matched_rows.append(row)

    # 4. 输出匹配结果
    print("\n以下是匹配的行：")
    matched_df = pd.DataFrame(matched_rows)
    print(matched_df)
    print(f"\n一共有 {matched_count} 行匹配。")


if __name__ == "__main__":
    # 修改为你的文件路径
    excel_file = "/home/kndhjk/PycharmProjects/cs773-2025-assignments-group-9/corner list for the tongariro_left_01 image.xlsx"
    csv_file = "/home/kndhjk/PycharmProjects/cs773-2025-assignments-group-9/left_image_corners.csv"

    compare_excel_csv(excel_file, csv_file)
    excel_file2 = "/home/kndhjk/PycharmProjects/cs773-2025-assignments-group-9/corner list for the tongariro_right_01 image.xlsx"
    csv_file2 = "/home/kndhjk/PycharmProjects/cs773-2025-assignments-group-9/right_image_corners.csv"
    compare_excel_csv(excel_file2, csv_file2)
