"""
 Flask-static-site is a skeleton Python/Flask application ready to
 be deployed as a static website. It is released under an MIT Licence.

 This file is the only Python script, and controls the entire app.
 Feel free to explore and adapt it to your own needs.

"""
import sys
from datetime import datetime

import sys
import classifier as clf
import local_config as lc
import os
from werkzeug.utils import secure_filename
from flask import Flask,request, redirect, render_template, render_template_string,url_for

from flask import Flask, render_template, render_template_string

from flask_frozen import Freezer
from flask_flatpages import (
    FlatPages, pygmented_markdown, pygments_style_defs)



def prerender_jinja(text):
    """ Pre-renders Jinja templates before markdown. """
    prerendered_body = render_template_string(text)
    return pygmented_markdown(prerendered_body)





DEBUG = True
FLATPAGES_AUTO_RELOAD = DEBUG
FLATPAGES_EXTENSION = '.md'
FLATPAGES_HTML_RENDERER = prerender_jinja
FLATPAGES_MARKDOWN_EXTENSIONS = ['codehilite']

app = Flask(__name__)
app.config.from_object(__name__)
pages = FlatPages(app)
freezer = Freezer(app)
app.config['UPLOAD_FOLDER'] = lc.OUTPUT_DIR




# === URL Routes === #

@app.route('/')
def index():
    page = pages.get_or_404('index')
    return render_template('pages/index.html', page=page)




@app.route('/<path:path>/')
def page(path):
    page = pages.get_or_404(path)
    return render_template('pages/page.html', page=page)


@app.route('/blog/')
def blog():
    articles = get_blog_articles(reverse=True)[:3]
    return render_template('blog/index.html', pages=articles)


@app.route('/blog/articles/')
def blog_articles():
    articles = get_blog_articles_year(reverse=True)
    return render_template('blog/articles.html', pages=articles)


@app.route('/blog/<path:path>/')
def blog_article(path):
    page = pages.get_or_404('blog/' + path)
    return render_template('blog/page.html', page=page)


@app.route('/pygments.css')
def pygments_css():
    return pygments_style_defs('tango'), 200, {'Content-Type': 'text/css'}



# === Blog articles === #

def get_blog_articles(reverse=False):
    """ Returns all published blog articles ordered by date. """
    articles = [p for p in pages if p.path.startswith('blog')]
    articles = [p for p in articles if p.meta.get('published', False)]
    articles = sorted(articles,
                      reverse=reverse,
                      key=lambda k: k.meta['date'])
    return articles


def get_blog_articles_year(reverse=False):
    """ Returns a list of articles indexed by year. """
    articles = []
    for article in get_blog_articles(reverse):
        year = article.meta['date'].year
        if len(articles) == 0 or articles[-1][0] != year:
            articles.append([year, [article]])
        else:
            articles[-1][1].append(article)
    return articles


@app.route('/upload_image',methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # Check if no file was submitted to the HTML form
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and clf.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            output = clf.make_prediction_csv(filename)
            result = {
                    'output': output,
        
                }
            return render_template('/pages/show.html', result=result)
      
# === Main function  === #

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'build':
        freezer.freeze()
    else:
        app.run(host='127.0.0.1', port=8000)
