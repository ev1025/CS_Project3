from flask import Blueprint

mbp1 = Blueprint('bp1',__name__,url_prefix='/bp1')

@mbp1.route('/')
def blue1():
    return "bp1 입니다."