import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit_toggle as tog
import numpy as np


@st.cache_resource
def load_data(data_location):
    data = pd.read_csv(data_location)
    return data

data_location = "class10_data_iit.csv"
with st.spinner('Preparing data...'):
    data = load_data(data_location)

if __name__=="__main__":
    #st.title('TN Academic Performance Dashboard 🏫🏫')
    new_title = '<p style=" font-size: 32px;">TN Academic Performance Dashboard 🏫🏫</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    col3, col4, col5= st.columns(3)
    with col3:
        year = st.multiselect('Academic Year:',
                              ['2018-2019', '2019-2020', '2020-2021', '2021-2022', '2022-2023'])
    with col4:
        std = st.multiselect('Standard:', ['8', '9', '10', '11', '12'])
    with col5:
        term = st.multiselect('Term:', ['All','1', '2', '3'])

    col1, col2, col3 = st.columns(3)
    with col1:
        city = st.multiselect('District', data['revname'].unique())
    with col2:
        branch = st.multiselect('School:', data.loc[(data['revname'].isin(city))]['sch_name'].unique())
    with col3:
        no_schools = st.slider("Number of performance:", min_value=1, max_value=10, step=1, value=5)

    col1, col2 = st.columns(2)
    with col1:
        top_students = st.checkbox('Student Info')
    with col2:
        top_performance = st.checkbox('Top Performance',value=True,help="Uncheck to get poor performance")

    #data_subset1 = data.loc[(data['revname'].isin(city))]

    if len(branch) > 1 and not top_students:


        data_subset = data.loc[(data['revname'].isin(city)) & (data['sch_name'].isin(branch))]
        if top_performance:
            new_title = f'<p style=" font-size: 28px;">Top {no_schools} schools</p>'

            st.markdown(new_title, unsafe_allow_html=True)
            #st.header()
            data_subset = data_subset.groupby('sch_name')['exam_mark'].median().sort_values(ascending=False).head(
                no_schools)
            fig = px.bar(
                data_subset, y=data_subset.index, x='exam_mark', orientation='h',
                labels={'sch_name': '', 'Percentage': 'Average marks'},
                # color_discrete_sequence=['#F63366']
                color="exam_mark",
                color_continuous_scale='Spectral',
            )
        else:
            new_title = f'<p style=" font-size: 28px;">Poor Performing {no_schools} schools</p>'

            st.markdown(new_title, unsafe_allow_html=True)
            #st.header(f'Poor Performing {no_schools} schools')
            data_subset = data_subset.groupby('sch_name')['exam_mark'].median().sort_values(ascending=True).head(
                no_schools)
            fig = px.bar(
                data_subset, y=data_subset.index, x='exam_mark', orientation='h',
                labels={'sch_name': '', 'Percentage': 'Average marks'},
                # color_discrete_sequence=['#F63366']
                color="exam_mark",
                color_continuous_scale='Oryel',
            )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig.update_layout(title_text='School names', title_x=0.5)

        st.plotly_chart(fig)

        st.caption('The graph shows the overall average exam marks of each school ranked from top to lowest')
    elif len(branch) > 1 and top_students:


        data_subset = data.loc[(data['revname'].isin(city)) & (data['sch_name'].isin(branch))]
        if top_performance:
            #st.header(f'Top {no_schools} students')
            new_title = f'<p style=" font-size: 28px;">Top {no_schools} students</p>'

            st.markdown(new_title, unsafe_allow_html=True)
            data_subset = data_subset.groupby('name').agg(
                {'exam_mark': 'median', 'sch_name': 'first', 'revname': 'first'}).sort_values(by='exam_mark',
                                                                                              ascending=False).head(
                no_schools)
            fig = px.bar(
                data_subset, y=data_subset.index, x='exam_mark', orientation='h',
                labels={'name': '', 'Percentage': 'Average marks'},
                # color_discrete_sequence=['#F63366']
                color="exam_mark",
                color_continuous_scale='Spectral',
            )

        else:
            new_title = f'<p style=" font-size: 28px;">Poor Performing {no_schools} students</p>'

            st.markdown(new_title, unsafe_allow_html=True)
            #st.header(f'Poor Performing {no_schools} students')
            data_subset = data_subset.groupby('name').agg(
                {'exam_mark': 'median', 'sch_name': 'first', 'revname': 'first'}).sort_values(by='exam_mark',
                                                                                              ascending=True).head(
                no_schools)
            fig = px.bar(
                data_subset, y=data_subset.index, x='exam_mark', orientation='h',
                labels={'name': '', 'Percentage': 'Average marks'},
                # color_discrete_sequence=['#F63366']
                color="exam_mark",
                color_continuous_scale='Oryel',
            )
        hovertemplate = ('Marks: %{x}<br>' +
                         'Name: %{y}<br>' +
                         'School: %{customdata[0]}<br>' +
                         'District: %{customdata[1]}' +
                         '<extra></extra>')
        fig.update_traces(customdata=np.stack((data_subset['sch_name'], data_subset['revname']), axis=-1),hovertemplate=hovertemplate)

        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig.update_layout(title_text='Student names', title_x=0.5)

        st.plotly_chart(fig)

        st.caption('The graph shows the overall top performers in selected filters from top to lowest')
    elif len(branch) == 1:


        data_subset = data.loc[(data['sch_name'] == branch[0])]
        if top_performance:
            #st.header(f'Top {no_schools} performers in \'{branch[0]}\' ')
            new_title = f'<p style=" font-size: 28px;">Top {no_schools} performers in \'{branch[0]}\' </p>'

            st.markdown(new_title, unsafe_allow_html=True)
            data_subset = data_subset.groupby('name').agg({'exam_mark': 'median','sch_name': 'first','revname': 'first'}).sort_values(by='exam_mark',ascending=False).head(
                no_schools)
            fig = px.bar(
                data_subset, y=data_subset.index, x='exam_mark', orientation='h',
                labels={'name': '', 'Percentage': 'Average marks'},
                # color_discrete_sequence=['#F63366']
                color="exam_mark",
                color_continuous_scale='Spectral',
            )

        else:
            #st.header(f'Poor {no_schools} performers in \'{branch[0]}\' ')
            new_title = f'<p style=" font-size: 28px;">Poor {no_schools} performers in \'{branch[0]}\'</p>'

            st.markdown(new_title, unsafe_allow_html=True)
            data_subset = data_subset.groupby('name').agg(
                {'exam_mark': 'median', 'sch_name': 'first', 'revname': 'first'}).sort_values(by='exam_mark',
                                                                                              ascending=True).head(
                no_schools)
            fig = px.bar(
                data_subset, y=data_subset.index, x='exam_mark', orientation='h',
                labels={'name': '', 'Percentage': 'Average marks'},
                # color_discrete_sequence=['#F63366']
                color="exam_mark",
                color_continuous_scale='Oryel',
            )
        hovertemplate = ('Marks: %{x}<br>' +
                         'Name: %{y}<br>' +
                         'School: %{customdata[0]}<br>' +
                         'District: %{customdata[1]}' +
                         '<extra></extra>')
        fig.update_traces(customdata=np.stack((data_subset['sch_name'], data_subset['revname']), axis=-1),hovertemplate=hovertemplate)
        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig.update_layout(title_text='Student names', title_x=0.5)

        st.plotly_chart(fig)

        st.caption('The graph shows the overall top performers in selected filters from top to lowest')

    elif len(branch) == 0 and len(city) > 0 and not top_students:


        data_subset = data.loc[(data['revname'].isin(city))]
        if top_performance:
            #st.header(f'Top {no_schools} schools')
            new_title = f'<p style=" font-size: 28px;">Top {no_schools} schools</p>'

            st.markdown(new_title, unsafe_allow_html=True)
            data_subset = data_subset.groupby('sch_name')['exam_mark'].median().sort_values(ascending=False).head(
                no_schools)
            fig = px.bar(
                data_subset, y=data_subset.index, x='exam_mark', orientation='h',
                labels={'sch_name': '', 'Percentage': 'Average marks'},
                # color_discrete_sequence=['#F63366']
                color="exam_mark",
                color_continuous_scale='Spectral',
            )
        else:
            #st.header(f'Poor Performing {no_schools} schools')
            new_title = f'<p style=" font-size: 28px;">Poor Performing {no_schools} schools</p>'

            st.markdown(new_title, unsafe_allow_html=True)
            data_subset = data_subset.groupby('sch_name')['exam_mark'].median().sort_values(ascending=True).head(
                no_schools)

            fig = px.bar(
                data_subset, y=data_subset.index, x='exam_mark', orientation='h',
                labels={'sch_name': '', 'Percentage': 'Average marks'},
                # color_discrete_sequence=['#F63366']
                color="exam_mark",
                color_continuous_scale='Oryel',
            )

        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig.update_layout(title_text='School names', title_x=0.5)

        st.plotly_chart(fig)

        st.caption('The graph shows the overall average exam marks of each school ranked from top to lowest')

    elif len(branch) == 0 and len(city) > 0 and top_students:


        data_subset = data.loc[(data['revname'].isin(city))]
        if top_performance:
            #st.header(f'Top {no_schools} students')
            new_title = f'<p style=" font-size: 28px;">Top {no_schools} students</p>'

            st.markdown(new_title, unsafe_allow_html=True)

            data_subset = data_subset.groupby('name').agg(
                {'exam_mark': 'median', 'sch_name': 'first', 'revname': 'first'}).sort_values(by='exam_mark',
                                                                                              ascending=False).head(
                no_schools)
            fig = px.bar(
                data_subset, y=data_subset.index, x='exam_mark', orientation='h',
                labels={'name': '', 'Percentage': 'Average marks'},
                # color_discrete_sequence=['#F63366']
                color="exam_mark",
                color_continuous_scale='Spectral',
            )
        else:
            #st.header(f'Poor Performing {no_schools} students')
            new_title = f'<p style=" font-size: 28px;">Poor Performing {no_schools} students</p>'

            st.markdown(new_title, unsafe_allow_html=True)
            data_subset = data_subset.groupby('name')['exam_mark'].median().sort_values(ascending=True).head(
                no_schools)
            fig = px.bar(
                data_subset, y=data_subset.index, x='exam_mark', orientation='h',
                labels={'name': '', 'Percentage': 'Average marks'},
                # color_discrete_sequence=['#F63366']
                color="exam_mark",
                color_continuous_scale='Oryel',
            )
        hovertemplate = ('Marks: %{x}<br>' +
                         'Name: %{y}<br>' +
                         'School: %{customdata[0]}<br>' +
                         'District: %{customdata[1]}' +
                         '<extra></extra>')
        fig.update_traces(customdata=np.stack((data_subset['sch_name'], data_subset['revname']), axis=-1),hovertemplate=hovertemplate)
        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig.update_layout(title_text='Student names', title_x=0.5)

        st.plotly_chart(fig)

        st.caption('The graph shows the overall top performers in selected filters from top to lowest')

    elif len(branch) == 0 and len(city) == 0 and not top_students:


        #data_subset = data.loc[(data['revname'].isin(city))]
        if top_performance:
            #st.header(f'Top {no_schools} schools')
            new_title = f'<p style=" font-size: 28px;">Top {no_schools} schools</p>'

            st.markdown(new_title, unsafe_allow_html=True)
            data_subset = data.groupby('sch_name')['exam_mark'].median().sort_values(ascending=False).head(
                no_schools)
            fig = px.bar(
                data_subset, y=data_subset.index, x='exam_mark', orientation='h',
                labels={'sch_name': '', 'Percentage': 'Average marks'},
                # color_discrete_sequence=['#F63366']
                color="exam_mark",
                color_continuous_scale='Spectral',
            )

        else:
            #st.header(f'Poor performing {no_schools} schools')
            new_title = f'<p style=" font-size: 28px;">Poor Performing {no_schools} schools</p>'

            st.markdown(new_title, unsafe_allow_html=True)
            data_subset = data.groupby('sch_name')['exam_mark'].median().sort_values(ascending=True).head(
                no_schools)

            fig = px.bar(
                data_subset, y=data_subset.index, x='exam_mark', orientation='h',
                labels={'sch_name': '', 'Percentage': 'Average marks'},
                #color_discrete_sequence=['#F63366']
                color="exam_mark",
                color_continuous_scale='Oryel',
            )
        #fig.update_yaxes(tickfont_family="Arial Black")

        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig.update_layout(title_text='School names', title_x=0.5)

        st.plotly_chart(fig)

        st.caption('The graph shows the overall top performing schools across all districts')
    elif len(branch) == 0 and len(city) == 0 and top_students:


        #data_subset = data.loc[(data['revname'].isin(city))]
        if top_performance:
            #st.header(f'Top {no_schools} students')
            new_title = f'<p style=" font-size: 28px;">Top {no_schools} student</p>'

            st.markdown(new_title, unsafe_allow_html=True)
            data_subset = data.groupby('name').agg(
                {'exam_mark': 'median', 'sch_name': 'first', 'revname': 'first'}).sort_values(by='exam_mark',
                                                                                              ascending=False).head(
                no_schools)
            fig = px.bar(
                data_subset, y=data_subset.index, x='exam_mark', orientation='h',
                labels={'name': '', 'Percentage': 'Average marks'},
                # color_discrete_sequence=['#F63366']
                color="exam_mark",
                color_continuous_scale='Spectral',
            )
        else:
            #st.header(f'Poor Performing {no_schools} students')
            new_title = f'<p style=" font-size: 28px;">Poor Performing {no_schools} students</p>'

            st.markdown(new_title, unsafe_allow_html=True)
            data_subset = data.groupby('name').agg(
                {'exam_mark': 'median', 'sch_name': 'first', 'revname': 'first'}).sort_values(by='exam_mark',
                                                                                              ascending=True).head(
                no_schools)

            fig = px.bar(
                data_subset, y=data_subset.index, x='exam_mark', orientation='h',
                labels={'name': '', 'Percentage': 'Average marks'},
                # color_discrete_sequence=['#F63366']
                color="exam_mark",
                color_continuous_scale='Oryel',
            )
        hovertemplate = ('Marks: %{x}<br>' +
                         'Name: %{y}<br>' +
                         'School: %{customdata[0]}<br>' +
                         'District: %{customdata[1]}' +
                         '<extra></extra>')
        fig.update_traces(customdata=np.stack((data_subset['sch_name'], data_subset['revname']), axis=-1),hovertemplate=hovertemplate)

        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig.update_layout(title_text='Student names', title_x=0.5)

        st.plotly_chart(fig)

        st.caption('The graph shows the overall top performing students across all districts')
    #print(data_subset)

st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
df1 = pd.read_csv("before_normalisation.csv")
df2 = pd.read_csv("after_normalisation.csv")

df = pd.read_csv("class10_data_iit.csv")

#st.title('Student Performance Dashboard')

new_title = '<p style=" font-size: 28px;">Missing value Imputation 🏫🏫</p>'
st.markdown(new_title, unsafe_allow_html=True)

st.subheader('Before Normalization')
st.dataframe(df1)
description = "Remarks: Students missing tough exams are getting higher average and are ranked in top position, this is putting other students at disadvantage. Students missing easy exams are penalized"
st.write(description)
st.subheader('After Normalization')
st.dataframe(df2)
description2 ="Remarks: Students missing tough exams are getting higher average is solved and students attending tough exams are not penalized and given fair chance for ranking"
st.write(description2)
col1, col2, col3 = st.columns(3)
with col1:
    tog.st_toggle_switch(label="Normalization of Missing Exam",
                        key="Key1",
                        default_value=False,
                        label_after = False,
                        inactive_color = '#D3D3D3',
                        active_color="#11567f",
                        track_color="#29B5E8"
                        )
with col2:
    weightage = st.slider("Consistency Weightage", min_value=1, max_value=100, step=1, value=50)
    # exam_weight = 100 - weightage
with col3:
    exam_att_weightage = st.slider("Exam Attendance Weightage", min_value=1, max_value=100, step=1, value=50)
st.subheader('Data Visualization of Revised Ranking')
left_column, right_column = st.columns(2)


df3 = pd.merge(df1, df2, on='Student ID',  how='left')
df3 = df3[['Student ID','Rank_x','Rank_y']]
#print(df)
fig = px.bar(df3, x="Student ID", y=["Rank_x", "Rank_y"], color_discrete_sequence=["indianred", "lightsalmon"], barmode='group')
texts = [df3['Rank_x'], df3['Rank_y']]
for i, t in enumerate(texts):
    fig.data[i].text = t
    fig.data[i].textposition = 'outside'
fig.update_layout(xaxis_title="Student ID",
                yaxis_title="Rank",
                barmode='group',
                bargap=0.15, # gap between bars of adjacent location coordinates.
                bargroupgap=0.1 # gap between bars of the same location coordinate.

                  )
# st.dataframe(df) # if need to display dataframe
st.plotly_chart(fig)

