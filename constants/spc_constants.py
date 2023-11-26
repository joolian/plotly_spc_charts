import pandas as pd
import re
""" Script used to extract factors for control charts from paper"""

def to_float(value):
    value = value.replace(' ', '')
    return value


def first_bit(page, start_row, stop_row):
    temp = page.iloc[start_row:stop_row]['c1'].str.split(expand=True, n=1)
    temp.columns = ['n', 'A']
    temp['A'] = temp['A'].str.replace(' ', '')
    temp = temp.astype({'n': int, 'A': float})
    return temp


def second_bit(page, start_row, stop_row):
    temp = page.iloc[start_row:stop_row, :1]
    temp = temp['c1'].str.extractall(r'(\d[.]\d{5}\s{1}\d{5})')
    temp = temp.unstack().T.reset_index(drop=True)
    temp.columns = ['A2', 'A3', 'B3', 'B4', 'B5', 'B6']
    temp = temp.map(lambda x: to_float(x))
    temp = temp.astype(float)
    return temp


def third_bit(page, start_row, stop_row):
    temp = ' ' + page.iloc[start_row:stop_row, :1].iloc[0,0]
    n = re.findall('(\s{1}\d{1,3}\s{1})', temp)
    n = [int(m.strip()) for m in n]
    values = re.findall(r'(\d[.]\d{5}\s{1}\d{5})', temp)
    values = [float(value.replace(' ', '')) for value in values]
    df = pd.DataFrame(data={'n': n, 'c4': values})
    return df


def fourth_bit(page, start_row, stop_row):
    temp = page.iloc[start_row:stop_row, :1]
    temp = temp['c1'].str.extractall(r'(\d[.]\d{5}\s{1}\d{5})')
    temp = temp.unstack().T.reset_index(drop=True)
    temp.columns = ['d2', 'd3', 'D1', 'D2', 'D3', 'D4', 'E2']
    temp = temp.map(lambda x: to_float(x))
    temp = temp.astype(float)
    return temp


if __name__ == '__main__':
    page_1 = pd.read_excel('control chart constants.xlsx', sheet_name='page_1')
    page_1.columns = ['c1']
    part_1 = first_bit(page_1, 0, 22)
    part_2 = second_bit(page_1, 22, 28)
    table_1 = pd.concat([part_1, part_2], axis=1)
    print(table_1)

    page_2 = pd.read_excel('control chart constants.xlsx', sheet_name='page_2')
    page_2.columns = ['c1']
    part_1 = first_bit(page_2, 0, 21)
    part_2 = second_bit(page_2, 21, 27)
    table_2 = pd.concat([part_1, part_2], axis=1)
    print(table_2)

    page_3 = pd.read_excel('control chart constants.xlsx', sheet_name='page_3')
    page_3.columns = ['c1']
    part_1 = first_bit(page_3, 0, 21)
    part_2 = second_bit(page_3, 21, 27)
    table_3 = pd.concat([part_1, part_2], axis=1)
    print(table_3)

    page_4 = pd.read_excel('control chart constants.xlsx', sheet_name='page_4')
    page_4.columns = ['c1']
    part_1 = first_bit(page_4, 0, 21)
    part_2 = second_bit(page_4, 21, 27)
    table_4 = pd.concat([part_1, part_2], axis=1)
    print(table_4)

    page_5 = pd.read_excel('control chart constants.xlsx', sheet_name='page_5')
    page_5.columns = ['c1']
    part_1 = third_bit(page_5, 0, 1)
    part_2 = second_bit(page_5, 1, 7)
    table_5 = pd.concat([part_1, part_2], axis=1)
    table_5 = table_5.rename(columns={'c4': 'A'})
    print(table_5)

    a_to_b6 = pd.concat([table_1, table_2, table_3, table_4, table_5]).reset_index(drop=True)

    page_6 = pd.read_excel('control chart constants.xlsx', sheet_name='page_6')
    page_6.columns = ['c1']
    part_1 = third_bit(page_6, 0, 1)
    part_2 = fourth_bit(page_6, 1, 8)
    table_6 = pd.concat([part_1, part_2], axis=1)
    print(table_6)

    page_7 = pd.read_excel('control chart constants.xlsx', sheet_name='page_7')
    page_7.columns = ['c1']
    part_1 = third_bit(page_7, 0, 1)
    part_2 = fourth_bit(page_7, 1, 8)
    table_7 = pd.concat([part_1, part_2], axis=1)
    print(table_7)

    page_8 = pd.read_excel('control chart constants.xlsx', sheet_name='page_8')
    page_8.columns = ['c1']
    part_1 = third_bit(page_8, 0, 1)
    part_2 = fourth_bit(page_8, 1, 8)
    table_8 = pd.concat([part_1, part_2], axis=1)
    print(table_8)

    page_9 = pd.read_excel('control chart constants.xlsx', sheet_name='page_9')
    page_9.columns = ['c1']
    part_1 = third_bit(page_9, 0, 1)
    part_2 = fourth_bit(page_9, 1, 8)
    table_9 = pd.concat([part_1, part_2], axis=1)
    print(table_9)

    page_10 = pd.read_excel('control chart constants.xlsx', sheet_name='page_10')
    page_10.columns = ['c1']
    part_1 = third_bit(page_10, 0, 1)
    part_2 = fourth_bit(page_10, 1, 8)
    table_10 = pd.concat([part_1, part_2], axis=1)
    print(table_10)

    page_11 = pd.read_excel('control chart constants.xlsx', sheet_name='page_11')
    page_11.columns = ['c1']
    part_1 = third_bit(page_11, 0, 1)
    part_2 = fourth_bit(page_11, 1, 8)
    table_11 = pd.concat([part_1, part_2], axis=1)
    print(table_11)

    c4_to_e2 = pd.concat([table_6, table_7, table_8, table_9, table_10, table_11]).reset_index(drop=True)
    c4_to_e2 = c4_to_e2.drop(columns='n')
    all_constants = pd.concat([a_to_b6, c4_to_e2], axis=1)
    all_constants.to_csv('factor_values_for_shewart_charts.csv', index=False)
