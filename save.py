import xlwt

workbook = xlwt.Workbook(encoding='utf-8')
booksheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
# 存第一行cell(1,1)和cell(1,2)
booksheet.write(0, 0, 34)
booksheet.write(0, 1, 38)
# 存第二行cell(2,1)和cell(2,2)
booksheet.write(1, 0, 36)
booksheet.write(1, 1, 39)
# 存一行数据
rowdata = [43, 56]
for i in range(len(rowdata)):
    booksheet.write(2, i, rowdata[i])
workbook.save('test_xlwt.xls')
rowdata2 = [1, 2, 3, 4, 5, 6]
rowdata2.append(1.1)
for i in range(len(rowdata2)):
    row = i // 2
    col = i % 2
    booksheet.write(row, col, rowdata2[i])
workbook.save('flower.xls')
