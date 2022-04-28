import unicodedata
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]
#
# usage python3 norm.py <input_file> <output_file>
#

dict_map = {
"òa": "oà",
"Òa": "Oà",
"ÒA": "OÀ",
"óa": "oá",
"Óa": "Oá",
"ÓA": "OÁ",
"ỏa": "oả",
"Ỏa": "Oả",
"ỎA": "OẢ",
"õa": "oã",
"Õa": "Oã",
"ÕA": "OÃ",
"ọa": "oạ",
"Ọa": "Oạ",
"ỌA": "OẠ",
"òe": "oè",
"Òe": "Oè",
"ÒE": "OÈ",
"óe": "oé",
"Óe": "Oé",
"ÓE": "OÉ",
"ỏe": "oẻ",
"Ỏe": "Oẻ",
"ỎE": "OẺ",
"õe": "oẽ",
"Õe": "Oẽ",
"ÕE": "OẼ",
"ọe": "oẹ",
"Ọe": "Oẹ",
"ỌE": "OẸ",
"ùy": "uỳ",
"Ùy": "Uỳ",
"ÙY": "UỲ",
"úy": "uý",
"Úy": "Uý",
"ÚY": "UÝ",
"ủy": "uỷ",
"Ủy": "Uỷ",
"ỦY": "UỶ",
"ũy": "uỹ",
"Ũy": "Uỹ",
"ŨY": "UỸ",
"ụy": "uỵ",
"Ụy": "Uỵ",
"ỤY": "UỴ",
}

def replace_all(text, dict_map):
  for i, j in dict_map.items():
    text = text.replace(i, j)
  return text


with open(input_file,'r',encoding='utf-8') as i:
  text = i.read()

text = unicodedata.normalize('NFKC',text)
normalized_text = replace_all(text, dict_map)

with open(output_file,'w',encoding='utf-8') as o:
  o.write(normalized_text)