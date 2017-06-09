import os

def normalize_to_unicode(text, encoding="utf-8"):
    import sys
    if sys.version_info.major == 2:
        if isinstance(text, str):
            text = unicode(text.decode(encoding))
        return text
    else:
        if isinstance(text, bytes):
            text = str(text.decode(encoding))
        return text

def normalize_to_bytes(text, encoding="utf-8"):
    import sys
    if sys.version_info.major == 2:
        if isinstance(text, unicode):
            text = str(text.encode(encoding))
        return text
    else:
        if isinstance(text, str):
            text = bytes(text.encode(encoding))
        return text

def convert_windows1252_to_utf8(text):
        #return text.decode("windows-1252").encode().decode("utf-8")
        return text.decode("cp1252")

def add_newline_to_unicode(text):
    return text + u"\n"

def single_item_process_standard(text):
    text = normalize_to_unicode(text).strip()
    text = add_newline_to_unicode(text)
    return normalize_to_bytes(text)

def single_item_process_funky(text):
    text = convert_windows1252_to_utf8(text).strip()
    text = add_newline_to_unicode(text)
    return normalize_to_bytes(text)


def process(lines):
    out_lines = []
    for line in lines:
        try:
            out_lines.append(single_item_process_standard(line))
        except UnicodeDecodeError:
            out_lines.append(single_item_process_funky(line))
    return out_lines
            
def open_process_rewrite(filename):
    with open(filename, "rb") as fp:
        lines = fp.readlines()
        #print 'lines: -------------'
        #print lines

    lines = process(lines)
    path_part, ext_part = os.path.splitext(filename)
    new_filename = "{}_processed{}".format(path_part, ext_part)
    with open(new_filename, "wb") as fp:
        fp.writelines(lines)

def best_practice_load(filename):
    out = []
    with open(filename, "rb") as fp:
        for line in fp.readlines():
            try:
                out.append(normalize_to_unicode(line).strip())
            except UnicodeDecodeError:
                print("Broken line: {}".format(line))
    return out

def best_practice_load_with_pandas(filename):
    import pandas as pd
    return pd.DataFrame.from_csv(filename)


if __name__ == '__main__':
    open_process_rewrite('test_bodies.csv')
    open_process_rewrite('test_stances_unlabeled.csv')
    open_process_rewrite('train_bodies.csv')
    open_process_rewrite('train_stances.csv')
