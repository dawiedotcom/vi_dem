import jinja2
import os
from jinja2 import Template
latex_jinja_env = jinja2.Environment(
    block_start_string = '\BLOCK{',
    block_end_string = '}',
    variable_start_string = '\VAR{',
    variable_end_string = '}',
    comment_start_string = '\#{',
    comment_end_string = '}',
    line_statement_prefix = '%%',
    line_comment_prefix = '%#',
    trim_blocks = True,
    autoescape = False,
    loader = jinja2.FileSystemLoader(os.path.abspath('.'))
)
#template = latex_jinja_env.get_template('jinja-test.tex')
#print(template.render(section1='Long Form', section2='Short Form'))

class Template:
    def __init__(self, filename):
        self.template = latex_jinja_env.get_template(filename)

    def render(self, *args, **kwargs):
        return self.template.render(*args, **kwargs)


if __name__ == '__main__':
    template = Template('figures/dem_E_vs_time_snapshot.tex_template')
    print(template.render(positions=[[10, 20]], d=5))
