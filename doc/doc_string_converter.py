"""

Author: Tranter Tech
Date: 2024
"""
import os
import re

_BASE_DIR = '../'
_INCLUDE = ['app/', 'comms/', 'utils/', 'app/driver_libs/jena.py', 'app/driver_libs/Xeryon.py',
            'optimise.py', 'preamble.py', 'main.py']


def get_folder_contents(folder):
    dir_list = os.listdir(_BASE_DIR + folder)
    removals = []
    for idx, dir_file in enumerate(dir_list):
        if '.py' not in dir_file or dir_file == '__init__.py':
            removals.append(idx)

    removals.reverse()
    for idx in removals:
        dir_list.pop(idx)

    return dir_list

def status_print(text, indent):
    print('\t'*indent + text)

def pull_doc_strings(filepath):
    doc_strings = []
    status_print(f'Opening file: {filepath}', 0)
    contents = []
    with open(filepath, 'r') as source_file:
        line = True
        while line:
            line = source_file.readline()
            contents.append(line)

    func_match_regex = r"([\s]{0,4})def\s([\_\w\d]+)\(([\w\d\=\{\}\,\s]+)\)"
    class_match_regex = r"class\s([\w\.\(\)]+)[\:]{1}"

    markers = []
    for lidx, line in enumerate(contents):
        # find a function
        fm = re.match(func_match_regex, line)
        if fm is not None:
            spaces, func_name, args = fm.groups()
            status_print(f'Found function: {func_name} - with params: {args}', len(spaces) // 4 + 1)
            markers.append((lidx, [func_name, args], (len(spaces) // 4 + 1)))

        # find the class
        m = re.match(class_match_regex, line)
        if m is not None:
            class_name = m.groups()[0]
            status_print(f'Found Class: {class_name}', 1)
            markers.append((lidx, class_name, 1))

    loc = ''
    for idx, marker_type, indent in markers:
        # if it's a class, check for a doc string or comment
        if type(marker_type) is str:
            next_line = contents[idx + 1]
            if '#' in next_line:
                doc_strings.append((marker_type, 1, next_line, 'class'))
            elif "\"\"\"" in next_line:
                if "\n" in next_line:
                    line_counter = 2
                    while (line_counter + idx) < len(contents):
                        new_line = contents[idx + line_counter]
                        next_line += new_line
                        line_counter += 1
                        if "\"\"\"" in new_line:
                            break
                doc_strings.append((marker_type, 1, next_line, 'class'))
            else:
                doc_strings.append((marker_type, 1, "", 'class'))

        elif type(marker_type) is list:
            args = marker_type[1].split(',')
            if args[0] == 'self':
                args.pop(0)
            args = [a.strip() for a in args]
            output_string = marker_type[0] + f"({', '.join(args)})"
            next_line = contents[idx + 1]
            if "\"\"\"" in next_line:
                if "\n" in next_line:
                    line_counter = 2
                    while (line_counter + idx) < len(contents):
                        new_line = contents[idx + line_counter]
                        next_line += new_line
                        line_counter += 1
                        if "\"\"\"" in new_line:
                            break
                doc_strings.append((output_string, indent, next_line, 'def'))
            else:
                doc_strings.append((output_string, indent, "", 'def'))

    return doc_strings

wrapper_prefix = lambda title: (r"\begin{tcolorbox}[enhanced jigsaw,breakable,pad at break*=1mm, colback=blue!5!white,colframe=blue!75!black,title="
                                + title + r",opacityback=0]\begin{flushleft}" + "\n")
wrapper_post = r"\end{flushleft}\end{tcolorbox}" + "\n"
line_break = r"\rule{\textwidth}{0.4pt}" + "\n"
title_wrap = lambda name, cmds, post: r"\textbf{" + name + "(}" + cmds + r"\textbf{):}\," + post + r"\vspace{0.2cm}\newline" + "\n"

def construct_text(files, doc_strings):
    output_string = ""
    for f, ds in zip(files, doc_strings):
        base_path = f[:-3].replace(r'/', ".").replace('_', r'\_')
        print('Compiling ' + base_path + ' ...')

        output_string += r"\section{" + f.replace('_', '\_') + "}\n"

        first_line = True
        terminated = True
        for d in ds:
            name, depth, doc_text, prefix = d

            # fix any naming issues
            name = name.replace('_', r'\_')

            if depth == 1:
                if not terminated:
                    output_string += wrapper_post

                output_string += wrapper_prefix(base_path + "." + name.split('(')[0])
                terminated = False
                first_line = True

                if doc_text != "":
                    doc_text = doc_text.strip().strip('\"\"\"')

                    doc_text = doc_text.strip()

                    doc_split = doc_text.split(':')

                    new_string = r"\emph{" + doc_split[0] + r"}\newline"
                    if len(doc_split) > 1:
                        new_string += "\n :" + ':'.join(doc_split[1:])

                    for m in re.findall(r":param\s[\w_]+:", new_string):
                        new_string = new_string.replace(m, "\n" + r"\textbf{" + m + "}")

                    new_string = new_string.replace(':return:', "\n" + r"\textbf{" + r":return:} ")

                    new_string = new_string.replace('_', r'\_')
                    new_string = new_string.replace('__', r'\_\_')

                    output_string += new_string + r"\newline"
                    output_string += line_break

            elif depth == 2:
                if not first_line:
                    output_string += line_break
                first_line = False

                func_name = name.split('(')[0]
                post_val = ''
                if func_name.strip()[:2] == r'\_' and func_name != r"\_\_init\_\_":
                    post_val = '[private]'

                args = name.split('(')[1][:-1]
                output_string += title_wrap(prefix + r"\," + func_name, args, post_val)

                if doc_text != "":
                    doc_text = doc_text.strip().strip('\"\"\"')

                    doc_text = doc_text.strip()

                    doc_split = doc_text.split(':')

                    new_string = r"\emph{" + doc_split[0] + r"}\newline"
                    if len(doc_split) > 1:
                        new_string += "\n :" + ':'.join(doc_split[1:])

                    for m in re.findall(r":param\s[\w_]+:", new_string):
                        new_string = new_string.replace(m, "\n" + r"\textbf{" + m + "}")

                    new_string = new_string.replace(':return:', "\n" + r"\textbf{" + r":return:} ")

                    new_string = new_string.replace('_', r'\_')
                    new_string = new_string.replace('__', r'\_\_')

                    output_string += new_string + r"\newline"
                else:
                    output_string += r"\emph{[ Inherited from base ]}\newline"

        output_string += wrapper_post

    return output_string

def main():
    # create the chapter file
    f = open('./chapters/3_ref_doc.tex', 'w+')
    f.write(r"\chapter{Reference Documentation}" + "\n")
    f.close()

    write_ops = [[], []]

    include_list = sorted(_INCLUDE)
    for dir_or_file in include_list:
        # skip it if it doesn't exist
        if not os.path.exists(_BASE_DIR + dir_or_file):
            print(f'ERROR: {_BASE_DIR + dir_or_file} does not exist.')
            continue

        # check if it's a file or folder
        is_file = os.path.isfile(_BASE_DIR + dir_or_file)

        # if it's not a file list the directory
        if not is_file:
            dir_list = get_folder_contents(dir_or_file)
            print(f'Found directory: {dir_or_file} - Files:\n{dir_list}')

            for file in dir_list:
                ds = pull_doc_strings(_BASE_DIR + dir_or_file + file)
                write_ops[0].append(dir_or_file + file)
                write_ops[1].append(ds)

    with open('./chapters/3_ref_doc.tex', 'a+') as tex_file:
        out_tex = construct_text(*write_ops)
        tex_file.write(out_tex)

                # raise NotImplementedError('Halt')


if __name__ == '__main__':
    main()