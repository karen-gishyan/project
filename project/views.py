from django.shortcuts import render


def display_graph(request,**kwargs):
    image_url=kwargs['image_url']
    return render(request,'index.html',context={"image_url":image_url})