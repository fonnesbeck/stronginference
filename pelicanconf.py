#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Christopher Fonnesbeck'
SITENAME = 'Strong Inference'
SITEURL = 'http://stronginference.com'

PATH = 'content'

TIMEZONE = 'America/Chicago'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
# FEED_ALL_ATOM = None
# CATEGORY_FEED_ATOM = None
# TRANSLATION_FEED_ATOM = None
# AUTHOR_FEED_ATOM = None
# AUTHOR_FEED_RSS = None
FEED_RSS = 'feeds/all.rss.xml'

# Blogroll
LINKS = (('Statistical Modeling, Causal Inference, and Social Science', 'http://andrewgelman.com/'),
         ('John D. Cook', 'http://www.johndcook.com/blog/'),
         ('Pythonic Perambulations', 'https://jakevdp.github.io'),
         ('Healthy Algorithms', 'http://healthyalgorithms.com'))

# Social widget
SOCIAL = (('Twitter', 'https://twitter.com/fonnesbeck'),
        ('Github', 'http://github.com/fonnesbeck'))

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

GITHUB_URL = 'http://github.com/fonnesbeck'
TWITTER_USERNAME = 'fonnesbeck'

PELICAN_SOBER_TWITTER_CARD_CREATOR = 'fonnesbeck'
# PELICAN_SOBER_ABOUT = 'Bayes, Python and Data'

MARKUP = ('md', 'ipynb')

PLUGIN_PATHS = ['./plugins']
PLUGINS = ['ipynb.markup']