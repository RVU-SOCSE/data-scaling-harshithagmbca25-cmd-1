Python 3.13.3 (tags/v3.13.3:6280bb5, Apr  8 2025, 14:47:33) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
import pandas as pd
 
df = pd.read_csv('C:/Users/Chandu/Downloads/10prog_4laptops.csv')
df
    Manufacturer       Model_Name  ... Weight_kg        Price
0          Apple      MacBook Pro  ...      1.37  11912523.48
1          Apple      Macbook Air  ...      1.34   7993374.48
2             HP           250 G6  ...      1.86   5112900.00
3          Apple      MacBook Pro  ...      1.83  22563005.40
4          Apple      MacBook Pro  ...      1.37  16037611.20
..           ...              ...  ...       ...          ...
972         Dell     Alienware 17  ...      4.42  24897600.00
973      Toshiba  Tecra A40-C-1DF  ...      1.95  10492560.00
974         Asus        Rog Strix  ...      2.73  18227710.80
975           HP      Probook 450  ...      2.04   8705268.00
976       Lenovo    ThinkPad T460  ...      1.70   8909784.00

[977 rows x 13 columns]
# Onehot encoding without sci-kit learn library
pd.get_dummies(df)
     Screen_size_inches  ...  Operating_System_Version_X
0                  13.3  ...                       False
1                  13.3  ...                       False
2                  15.6  ...                       False
3                  15.4  ...                       False
4                  13.3  ...                       False
..                  ...  ...                         ...
972                17.3  ...                       False
973                14.0  ...                       False
974                17.3  ...                       False
975                15.6  ...                       False
976                14.0  ...                       False

[977 rows x 811 columns]
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder()
enc_data = enc.fit_transform(df[['Category']])
enc_data
<Compressed Sparse Row sparse matrix of dtype 'float64'
	with 977 stored elements and shape (977, 6)>
# Identifying the list of categories, one-hot encoding is considering as columns
enc.categories_
[array(['2 in 1 Convertible', 'Gaming', 'Netbook', 'Notebook', 'Ultrabook',
       'Workstation'], dtype=object)]
enc_data.toarray()
array([[0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 1., 0., 0.],
       ...,
       [0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0., 0.]], shape=(977, 6))
# creating a dataframe without giving column names
enc_df = pd.DataFrame(enc_data.toarray())
enc_df
       0    1    2    3    4    5
0    0.0  0.0  0.0  0.0  1.0  0.0
1    0.0  0.0  0.0  0.0  1.0  0.0
2    0.0  0.0  0.0  1.0  0.0  0.0
3    0.0  0.0  0.0  0.0  1.0  0.0
4    0.0  0.0  0.0  0.0  1.0  0.0
..   ...  ...  ...  ...  ...  ...
972  0.0  1.0  0.0  0.0  0.0  0.0
973  0.0  0.0  0.0  1.0  0.0  0.0
974  0.0  1.0  0.0  0.0  0.0  0.0
975  0.0  0.0  0.0  1.0  0.0  0.0
976  0.0  0.0  0.0  1.0  0.0  0.0

[977 rows x 6 columns]
# creating a dataframe with giving column names
enc_df = pd.DataFrame(enc_data.toarray(), columns = ['2 in 1 Convertible',
'Gaming', 'Netbook', 'Notebook', 'Ultrabook', 'Workstation'])
enc_df
     2 in 1 Convertible  Gaming  Netbook  Notebook  Ultrabook  Workstation
0                   0.0     0.0      0.0       0.0        1.0          0.0
1                   0.0     0.0      0.0       0.0        1.0          0.0
2                   0.0     0.0      0.0       1.0        0.0          0.0
3                   0.0     0.0      0.0       0.0        1.0          0.0
4                   0.0     0.0      0.0       0.0        1.0          0.0
..                  ...     ...      ...       ...        ...          ...
972                 0.0     1.0      0.0       0.0        0.0          0.0
973                 0.0     0.0      0.0       1.0        0.0          0.0
974                 0.0     1.0      0.0       0.0        0.0          0.0
975                 0.0     0.0      0.0       1.0        0.0          0.0
976                 0.0     0.0      0.0       1.0        0.0          0.0

[977 rows x 6 columns]
>>> df1 = df.join(enc_df)
>>> KeyboardInterrupt
>>> df1
    Manufacturer       Model_Name   Category  ...  Notebook Ultrabook Workstation
0          Apple      MacBook Pro  Ultrabook  ...       0.0       1.0         0.0
1          Apple      Macbook Air  Ultrabook  ...       0.0       1.0         0.0
2             HP           250 G6   Notebook  ...       1.0       0.0         0.0
3          Apple      MacBook Pro  Ultrabook  ...       0.0       1.0         0.0
4          Apple      MacBook Pro  Ultrabook  ...       0.0       1.0         0.0
..           ...              ...        ...  ...       ...       ...         ...
972         Dell     Alienware 17     Gaming  ...       0.0       0.0         0.0
973      Toshiba  Tecra A40-C-1DF   Notebook  ...       1.0       0.0         0.0
974         Asus        Rog Strix     Gaming  ...       0.0       0.0         0.0
975           HP      Probook 450   Notebook  ...       1.0       0.0         0.0
976       Lenovo    ThinkPad T460   Notebook  ...       1.0       0.0         0.0

[977 rows x 19 columns]
>>> LabelEncoder
Traceback (most recent call last):
  File "<pyshell#21>", line 1, in <module>
    LabelEncoder
NameError: name 'LabelEncoder' is not defined
>>> # Using label encoding
>>> from sklearn.preprocessing import LabelEncoder
>>> le = LabelEncoder()
>>> df['RAM'] = le.fit_transform(df['RAM'])
>>> df
    Manufacturer       Model_Name  ... Weight_kg        Price
0          Apple      MacBook Pro  ...      1.37  11912523.48
1          Apple      Macbook Air  ...      1.34   7993374.48
2             HP           250 G6  ...      1.86   5112900.00
3          Apple      MacBook Pro  ...      1.83  22563005.40
4          Apple      MacBook Pro  ...      1.37  16037611.20
..           ...              ...  ...       ...          ...
972         Dell     Alienware 17  ...      4.42  24897600.00
973      Toshiba  Tecra A40-C-1DF  ...      1.95  10492560.00
974         Asus        Rog Strix  ...      2.73  18227710.80
975           HP      Probook 450  ...      2.04   8705268.00
976       Lenovo    ThinkPad T460  ...      1.70   8909784.00

[977 rows x 13 columns]
>>> le.classes_
array(['12GB', '16GB', '24GB', '2GB', '32GB', '4GB', '6GB', '8GB'],
      dtype=object)
