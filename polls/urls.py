from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    # url(r'^handle$', infohandle.handle, name='index')
]

# urlpatterns = patterns("",
# 	(r'^search-form/$', search.search_form),
# 	(r'^search/$', search.search),
# )