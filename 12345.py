from bs4 import BeautifulSoup

html_code="""<!DOCTYPE html>
<html>
<head>
    <title>Sample Website</title>
</head>
<body>
<h2>HTML 연습!</h2>

<p>이것은 첫 번째 문단입니다.</p>
<p>이것은 두 번째 문단입니다!</p>

<ul>
    <li>커피</li>
    <li>녹차</li>
    <li>우유</li>
</ul>

<img src='https://i.imgur.com/bY0l0PC.jpg' alt="coffee"/>
<img src='https://i.imgur.com/fvJLWdV.jpg' alt="green-tea"/>
<img src='https://i.imgur.com/rNOIbNt.jpg' alt="milk"/>

</body>
</html>"""


soup=BeautifulSoup(html_code,'html.parser')

img_tags=soup.select('img')


b=[]


for i in img_tags:
    b.append(i['alt'])



print(b)