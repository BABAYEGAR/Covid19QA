import attr


@attr.s
class Answer(object):
    text = attr.ib()
    start_score = attr.ib()
    end_score = attr.ib()
    input_text = attr.ib()



