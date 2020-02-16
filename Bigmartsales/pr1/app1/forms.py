from django import forms
class QuestionForm(forms.Form):
   
    field1 = forms.CharField(label='field1', max_length = 50)
    field2 = forms.DecimalField(label='field2')
    field3 = forms.CharField(label='field3', max_length = 50)
    field4 = forms.DecimalField(label='field4')
    field5 = forms.CharField(label='field5', max_length = 50)
    field6 = forms.DecimalField(label='field6')
    field7 = forms.CharField(label='field7', max_length = 50)
    field8 = forms.IntegerField(label='field8')
    field9 = forms.CharField(label='field9', max_length = 50)
    field10 = forms.CharField(label='field10', max_length = 50)
    field11 = forms.CharField(label='field11', max_length = 50)
    
    
   
    
    