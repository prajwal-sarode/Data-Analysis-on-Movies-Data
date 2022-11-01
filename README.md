# Data-Analysis-on-Movies-Data

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.io as pyo
pyo.renderers.default='iframe'
import plotly.express as px
%matplotlib inline
```


```python
df = pd.read_csv('moviestreams.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>ID</th>
      <th>Title</th>
      <th>Year</th>
      <th>Age</th>
      <th>IMDb</th>
      <th>Rotten Tomatoes</th>
      <th>Netflix</th>
      <th>Hulu</th>
      <th>Prime Video</th>
      <th>Disney+</th>
      <th>Type</th>
      <th>Directors</th>
      <th>Genres</th>
      <th>Country</th>
      <th>Language</th>
      <th>Runtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>Inception</td>
      <td>2010</td>
      <td>13+</td>
      <td>8.8</td>
      <td>87%</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Christopher Nolan</td>
      <td>Action,Adventure,Sci-Fi,Thriller</td>
      <td>United States,United Kingdom</td>
      <td>English,Japanese,French</td>
      <td>148.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>The Matrix</td>
      <td>1999</td>
      <td>18+</td>
      <td>8.7</td>
      <td>87%</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Lana Wachowski,Lilly Wachowski</td>
      <td>Action,Sci-Fi</td>
      <td>United States</td>
      <td>English</td>
      <td>136.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>Avengers: Infinity War</td>
      <td>2018</td>
      <td>13+</td>
      <td>8.5</td>
      <td>84%</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Anthony Russo,Joe Russo</td>
      <td>Action,Adventure,Sci-Fi</td>
      <td>United States</td>
      <td>English</td>
      <td>149.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>Back to the Future</td>
      <td>1985</td>
      <td>7+</td>
      <td>8.5</td>
      <td>96%</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Robert Zemeckis</td>
      <td>Adventure,Comedy,Sci-Fi</td>
      <td>United States</td>
      <td>English</td>
      <td>116.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>The Good, the Bad and the Ugly</td>
      <td>1966</td>
      <td>18+</td>
      <td>8.8</td>
      <td>97%</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Sergio Leone</td>
      <td>Western</td>
      <td>Italy,Spain,West Germany</td>
      <td>Italian</td>
      <td>161.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (16744, 17)




```python
cols = df.columns.tolist()
cols
```




    ['Unnamed: 0',
     'ID',
     'Title',
     'Year',
     'Age',
     'IMDb',
     'Rotten Tomatoes',
     'Netflix',
     'Hulu',
     'Prime Video',
     'Disney+',
     'Type',
     'Directors',
     'Genres',
     'Country',
     'Language',
     'Runtime']




```python
df.drop(['Unnamed: 0', 'ID'], axis = 1, inplace=True )

cols = df.columns.tolist()

cols
```




    ['Title',
     'Year',
     'Age',
     'IMDb',
     'Rotten Tomatoes',
     'Netflix',
     'Hulu',
     'Prime Video',
     'Disney+',
     'Type',
     'Directors',
     'Genres',
     'Country',
     'Language',
     'Runtime']



## Checking For Missing Data


```python
df.isna().sum()
```




    Title                  0
    Year                   0
    Age                 9390
    IMDb                 571
    Rotten Tomatoes    11586
    Netflix                0
    Hulu                   0
    Prime Video            0
    Disney+                0
    Type                   0
    Directors            726
    Genres               275
    Country              435
    Language             599
    Runtime              592
    dtype: int64




```python
df.dtypes
```




    Title               object
    Year                 int64
    Age                 object
    IMDb               float64
    Rotten Tomatoes     object
    Netflix              int64
    Hulu                 int64
    Prime Video          int64
    Disney+              int64
    Type                 int64
    Directors           object
    Genres              object
    Country             object
    Language            object
    Runtime            float64
    dtype: object




```python
df['Age']
```




    0        13+
    1        18+
    2        13+
    3         7+
    4        18+
            ... 
    16739    NaN
    16740     7+
    16741    NaN
    16742    NaN
    16743    NaN
    Name: Age, Length: 16744, dtype: object




```python
age_map = {'18+':18,'7+':7,'13+':13,'All':0,'16+':16}

df['AgeCopy']  = df['Age'].map(age_map)

df['AgeCopy']
```




    0        13.0
    1        18.0
    2        13.0
    3         7.0
    4        18.0
             ... 
    16739     NaN
    16740     7.0
    16741     NaN
    16742     NaN
    16743     NaN
    Name: AgeCopy, Length: 16744, dtype: float64




```python
df['new_Rotten_Tomatoes'] = df['Rotten Tomatoes'].str.replace('%','')



for i in df['new_Rotten_Tomatoes']:
    if i == str:
        i.astype(int)
# OR
# df['new_Rotten_Tomatoes'] = df['new_Rotten Tomatoes'].astype(int)

```

# Visualisation

## Q. What is the number of MOVIES for each age group?


```python
df['Age'].value_counts()
```




    18+    3474
    7+     1462
    13+    1255
    all     843
    16+     320
    Name: Age, dtype: int64



## Q. Top 10 languages in Streaming Services


```python
df.Language.value_counts()
```




    English                                                                             10955
    Hindi                                                                                 503
    English,Spanish                                                                       276
    Spanish                                                                               267
    English,French                                                                        174
                                                                                        ...  
    English,German,Hungarian,Romanian                                                       1
    English,Spanish,Chinese,Latin                                                           1
    English,Danish,Malay,Dutch,Indonesian,Finnish,Luxembourgish,French Sign Language        1
    Dutch,French                                                                            1
    English,Algonquin                                                                       1
    Name: Language, Length: 1102, dtype: int64




```python
language = df.Language.value_counts().head(10)

plt.figure(figsize=(15,8))
plt.title('Top 10 Languages in Streaming Services')
sns.barplot(x=language.index,y=language.values)
```




    <AxesSubplot:title={'center':'Top 10 Languages in Streaming Services'}>




    
![png](output_16_1.png)
    



```python
from IPython.display import HTML
import plotly.express as px

fig = px.pie(df,
           values=language.values,
           names=language.index,
           title='Top 10 Languages in Streaming Services')

HTML(fig.to_html())
fig.show()
```


<iframe
    scrolling="no"
    width="100%"
    height="545px"
    src="iframe_figures/figure_14.html"
    frameborder="0"
    allowfullscreen
></iframe>



## Q. Numbers of movies in specific age group in All services


```python
plt.figure(figsize=(15,8))
plt.pie(df['AgeCopy'].value_counts().index,labels=df['AgeCopy'].value_counts(),autopct='%.0f%%')
plt.show()

```


    
![png](output_19_0.png)
    



```python
from IPython.display import HTML
import plotly.express as px


fig = px.bar(df,
           x=df['Age'].value_counts().index,
           y=df['Age'].value_counts(),
           title="Number of Movies in speccific age group in ALL Services",
           text=df['Age'].value_counts(),
           height=600)
fig.update_traces(texttemplate='%{text:.2s}',textposition='outside')
HTML(fig.to_html())
fig.show()
```


<iframe
    scrolling="no"
    width="100%"
    height="620"
    src="iframe_figures/figure_16.html"
    frameborder="0"
    allowfullscreen
></iframe>



## Q. Number of movies in specific age group in Netflix


```python
from IPython.display import HTML
import plotly.express as px
 
netflix_df=df[df['Netflix']==1]
    
fig = px.bar(netflix_df,
           x=netflix_df['Age'].value_counts().index,
           y=netflix_df['Age'].value_counts(),
           title="Number of Movies in speccific age group in Netflix",
           text=netflix_df['Age'].value_counts(),
           height=600)
fig.update_traces(texttemplate='%{text:.2s}',textposition='outside')
HTML(fig.to_html())
fig.show()
```


<iframe
    scrolling="no"
    width="100%"
    height="620"
    src="iframe_figures/figure_17.html"
    frameborder="0"
    allowfullscreen
></iframe>



## Q. Number of movies in specific age group in Amazon Prime


```python
from IPython.display import HTML
import plotly.express as px
 
prime_df=df[df['Prime Video']==1]
    
fig = px.bar(prime_df,
           x=prime_df['Age'].value_counts().index,
           y=prime_df['Age'].value_counts(),
           title="Number of Movies in speccific age group in Amazon Prime",
           text=prime_df['Age'].value_counts(),
           height=600)
fig.update_traces(texttemplate='%{text:.2s}',textposition='outside')
HTML(fig.to_html())
fig.show()
```


<iframe
    scrolling="no"
    width="100%"
    height="620"
    src="iframe_figures/figure_18.html"
    frameborder="0"
    allowfullscreen
></iframe>



## Q. Number of movies in specific age group in Disney+


```python
from IPython.display import HTML
import plotly.express as px
 
disney_df=df[df['Disney+']==1]
    
fig = px.bar(disney_df,
           x=disney_df['Age'].value_counts().index,
           y=disney_df['Age'].value_counts(),
           title="Number of Movies in speccific age group in Disney+",
           text=disney_df['Age'].value_counts(),
           height=600)
fig.update_traces(texttemplate='%{text:.2s}',textposition='outside')
HTML(fig.to_html())
fig.show()
```


<iframe
    scrolling="no"
    width="100%"
    height="620"
    src="iframe_figures/figure_19.html"
    frameborder="0"
    allowfullscreen
></iframe>



## Q. Number of movies in specific age group in Hulu


```python
from IPython.display import HTML
import plotly.express as px
 
hulu_df=df[df['Hulu']==1]
    
fig = px.bar(hulu_df,
           x=hulu_df['Age'].value_counts().index,
           y=hulu_df['Age'].value_counts(),
           title="Number of Movies in speccific age group in Hulu",
           text=hulu_df['Age'].value_counts(),
           height=600)
fig.update_traces(texttemplate='%{text:.2s}',textposition='outside')
HTML(fig.to_html())
fig.show()
```


<iframe
    scrolling="no"
    width="100%"
    height="620"
    src="iframe_figures/figure_20.html"
    frameborder="0"
    allowfullscreen
></iframe>



# Rotten Tomatoes Score

## Q. Rotten tomato ratimg for All Services


```python
from IPython.display import HTML
import plotly.express as px
     
fig = px.bar(df,
           x=df['Rotten Tomatoes'].value_counts().index,
           y=df['Rotten Tomatoes'].value_counts(),
           title="Overall Rotten Tomatoes Rating",
           text=df['Rotten Tomatoes'].value_counts(),
           height=600)
fig.update_traces(texttemplate='%{text:.2s}',textposition='outside')
HTML(fig.to_html())
fig.show()
```


<iframe
    scrolling="no"
    width="100%"
    height="620"
    src="iframe_figures/figure_21.html"
    frameborder="0"
    allowfullscreen
></iframe>



## Q. Rotten tomato ratimg for Each Services


```python
rt_scores=pd.DataFrame({'Streaming Service' :['Prime Video','Hulu','Disney+','Netflix'],
 'Rotten Tomatoes Score':[netflix_df['Rotten Tomatoes'].value_counts()[0],
                          hulu_df['Rotten Tomatoes'].value_counts()[0],
                          disney_df['Rotten Tomatoes'].value_counts()[0],
                          prime_df['Rotten Tomatoes'].value_counts()[0]
                      
     ]})

rt_scores.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Streaming Service</th>
      <th>Rotten Tomatoes Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Prime Video</td>
      <td>130</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hulu</td>
      <td>18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Disney+</td>
      <td>19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Netflix</td>
      <td>257</td>
    </tr>
  </tbody>
</table>
</div>




```python
sort_rt_scores=rt_scores.sort_values(ascending=False,by='Rotten Tomatoes Score')

sort_rt_scores
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Streaming Service</th>
      <th>Rotten Tomatoes Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Netflix</td>
      <td>257</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Prime Video</td>
      <td>130</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Disney+</td>
      <td>19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hulu</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = px.bar(sort_rt_scores,
           x=sort_rt_scores['Streaming Service'],
           y=sort_rt_scores['Rotten Tomatoes Score'],
           title="Rotten Tomatoes Rating For Each Service",
           text=sort_rt_scores['Rotten Tomatoes Score'],
           height=600)
fig.update_traces(marker_color='purple' ,texttemplate='%{text:.2s}',textposition='outside')
HTML(fig.to_html())
fig.show()
```


<iframe
    scrolling="no"
    width="100%"
    height="620"
    src="iframe_figures/figure_24.html"
    frameborder="0"
    allowfullscreen
></iframe>



# IMDB Ratings


```python
fig = px.bar(df,
           x=df['IMDb'].value_counts().index,
           y=df['IMDb'].value_counts(),
           title="Overall IMDb Ratings",
           text=df['IMDb'].value_counts(),
           height=600)
fig.update_traces(marker_color='red' ,texttemplate='%{text:.2s}',textposition='outside')
HTML(fig.to_html())
fig.show()
```


<iframe
    scrolling="no"
    width="100%"
    height="620"
    src="iframe_figures/figure_25.html"
    frameborder="0"
    allowfullscreen
></iframe>



## Count of Runtime of Movies


```python
RuntimeCount = pd.DataFrame(dict(df['Runtime'].value_counts().sort_values(ascending=False)[:10]).items(),
               columns=['Runtime','Count'])
```


```python
RuntimeCount
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Runtime</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90.0</td>
      <td>971</td>
    </tr>
    <tr>
      <th>1</th>
      <td>95.0</td>
      <td>489</td>
    </tr>
    <tr>
      <th>2</th>
      <td>92.0</td>
      <td>434</td>
    </tr>
    <tr>
      <th>3</th>
      <td>93.0</td>
      <td>422</td>
    </tr>
    <tr>
      <th>4</th>
      <td>85.0</td>
      <td>408</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>152</th>
      <td>19.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>153</th>
      <td>32.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>154</th>
      <td>9.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>155</th>
      <td>7.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>156</th>
      <td>10.0</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>157 rows × 2 columns</p>
</div>




```python
fig = px.bar(df,
           x=RuntimeCount['Runtime'],
           y=RuntimeCount['Count'],
           title="Count of RunTime of Movies",
           text=RuntimeCount['Runtime',
           height=600)
fig.update_traces(marker_color='black' ,texttemplate='%{text:.2s}',textposition='outside')
HTML(fig.to_html())
fig.show()
```


      File "C:\Users\prajw\AppData\Local\Temp\ipykernel_19316\1154088041.py", line 6
        height=600)
              ^
    SyntaxError: invalid syntax
    


## Directors and their cout of movies they have directed


```python
new_data=df[df['Directors'] !=np.nan]

directors_count=dict()

direc_in_data=list(new_data['Directors'].astype(str))

for xdir in direc_in_data:
    curr_dirs=xdir.split(',')
    for xd in curr_dirs:
        if xd in directors_count.keys():
            directors_count[xd]=directors_count.get(xd)+1
        else:
            directors_count[xd]=1
```


```python
DirCount=pd.DataFrame(directors_count.items(),columns=['Director','Count'])

DirCount=DirCount.sort_values(by='Count', ascending=False).head(20)

DirCount
```


```python
DirCount=DirCount.drop(56,axis=0)

DirCount
```


```python
fig = px.bar(DirCount,
           x=DirCount['Director'],
           y=DirCount['Count'],
           title="Directors and their count of movies they have directed",
           text=DirCount['Director'],
           height=600)
fig.update_traces(marker_color='purple' ,texttemplate='%{text:.2s}',textposition='outside')
HTML(fig.to_html())
fig.show()
```


```python
df[df['Directors']=='Jay Chapman'][['Directors','Title','Genres','Runtime']]
```

# Exploring Genres


```python
genres_=dict(df['Genres'].value_counts())

genres_count=dict()

for g,count in genres_.items():
    g=g.split(',')
    for i in g:
        if i in genres_count.keys():
            genres_count[i]=genres_count.get(i)+1
        else:
            genres_count[i]=1
```


```python
genres_count
```


```python
count_genres_df=pd.DataFrame(genres_count.items(),columns=['Genre','Count'])

count_genres_df=count_genres_df.sort_values(by='Count', ascending=False).head(20)

count_genres_df
```


```python
fig = px.bar(count_genres_df,
           x=count_genres_df['Genre'],
           y=count_genres_df['Count'],
           title="Genres and their count",
           text=count_genres_df['Count'],
           height=600)
fig.update_traces(marker_color='lightsalmon' ,texttemplate='%{text:.2s}',textposition='outside')
HTML(fig.to_html())
fig.show()
```

## Q. What are the top Movies on Each Platform?

### NetFlix


```python
data_netflix_top=netflix_df[netflix_df['IMDb']>8.5]

data_netflix_top=data_netflix_top[['Title','IMDb']].sort_values(ascending=False,by='IMDb')

data_netflix_top
```


```python
fig = px.bar(data_netflix_top,
           x=data_netflix_top['Title'],
           y=data_netflix_top['IMDb'],
           title="Top Movies on Netflix by IMDb",
           text=data_netflix_top['IMDb'],
           height=800)
fig.update_traces(marker_color='brown' ,texttemplate='%{text:.2s}',textposition='outside')
HTML(fig.to_html())
fig.show()
```

## Amazon Prime


```python
amz_top = prime_df[prime_df['IMDb']>8.5]
amz_top = amz_top[['Title', 'IMDb']].sort_values(ascending=False, by='IMDb')
amz_top
```


```python
fig = px.bar(amz_top, 
             x=amz_top['Title'], 
             y=amz_top['IMDb'],
             title="Top Movies On Amazon Prime",
             text=amz_top['IMDb'],
             height=800)
fig.update_traces(marker_color='brown',texttemplate='%{text:.2s}', textposition='outside') #for the text to be outside.
HTML(fig.to_html())
fig.show()
```

## On Disney+


```python
disney_top = disney_df[disney_df['IMDb']>8.5]
disney_top = disney_top[['Title', 'IMDb']].sort_values(ascending=False, by='IMDb')
disney_top
```


```python
fig = px.bar(disney_top, 
             x=disney_top['Title'], 
             y=disney_top['IMDb'],
             title="Top Movies On Disney+",
             text=disney_top['IMDb'],
             height=800)
fig.update_traces(marker_color='lightblue',texttemplate='%{text:.2s}', textposition='outside') #for the text to be outside.
HTML(fig.to_html())
fig.show()
```

## On HULU


```python
hulu_top = hulu_df[hulu_df['IMDb']>8.5]
hulu_top = hulu_top[['Title', 'IMDb']].sort_values(ascending=False, by='IMDb')
hulu_top
```


```python
fig = px.bar(hulu_top, 
             x=hulu_top['Title'], 
             y=hulu_top['IMDb'],
             title="Top Movies On Hulu",
             text=hulu_top['IMDb'],
             height=800)
fig.update_traces(marker_color='purple',texttemplate='%{text:.2s}', textposition='outside') #for the text to be outside.
HTML(fig.to_html())
fig.show()
```


```python

```
