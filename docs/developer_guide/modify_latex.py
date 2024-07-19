#!/usr/bin/python
# -*- coding: UTF-8

import os
import json
import re
import codecs
import shutil


class ModifyLatex:
    def __init__(self, content, json_dict):
        self.content = content
        self.tables = json_dict["tables"]
        self.json_dict = json_dict

    def replace_package(self):
        replace_dict = self.json_dict["replacepackage"]
        for key, value in replace_dict.items():
            self.content = re.sub(key, value, self.content, 0, re.M | re.I | re.U)
        return

    def modify_table_attributes(self):
        newcontent = self.content
        searchstr = r"(\\begin{savenotes}\\sphinxattablestart|\\begin{savenotes}\\sphinxatlongtablestart)([\s\S]*?)(\\sphinxattableend\\end{savenotes}|\\sphinxatlongtableend\\end{savenotes})"
        m = re.finditer(searchstr, self.content, re.M | re.I | re.U)
        for match in m:
            oldtablestr = match.group()
            tablestr = match.groups()
            caption_dict = {}
            for item in self.tables["styles"]:
                caption_dict.update({item["caption"]: item})
            if len(caption_dict) > 0:
                newtableattr = self._modify_single_table_attributes(
                    tablestr[0] + tablestr[1] + tablestr[2], caption_dict
                )
                newcontent = newcontent.replace(
                    tablestr[0] + tablestr[1] + tablestr[2], newtableattr
                )

        self.content = newcontent

    def _modify_single_table_attributes(self, singletablecontent, caption_dict):
        new_singletablecontent = singletablecontent
        if self.tables["isname"]:
            searchstr = r".*\\label.*?:(?P<caption>[\s\S].*)}}.*"
        else:
            searchstr = r"(\\sphinxcaption|\\caption){(?P<caption>[\s\S]*?)}"

        matchcaption = re.search(searchstr, singletablecontent, re.M | re.I | re.U)
        if matchcaption != None:
            tablecaption = matchcaption.group("caption")
        else:
            tablecaption = ""
        if tablecaption in caption_dict:
            tablestyle_dict = caption_dict[tablecaption]
            new_singletablecontent = self._start_modify_table_attr(
                singletablecontent,
                tablestyle_dict["isLongTable"],
                tablestyle_dict["isCusHead"],
            )
            if tablestyle_dict["isVertical"] == True:
                new_singletablecontent = self._modify_vertical_table(
                    new_singletablecontent
                )
        else:
            new_singletablecontent = self._start_modify_table_attr(
                singletablecontent, False
            )
        if new_singletablecontent == "":
            new_singletablecontent = singletablecontent

        return new_singletablecontent

    def _start_modify_table_attr(self, singletablecontent, islongtable, isCusHead=True):
        searchstr = r"(\\begin{tabular}|\\begin{tabulary})(\[[a-z]\]|{\\linewidth}\[[a-z]\])([\s\S].*)"
        splittable = re.split(searchstr, singletablecontent, 0, re.M | re.I | re.U)
        if splittable == None or len(splittable) < 5:
            searchstr = r"\\begin{longtable}([\s\S].*)"
            splittable = re.split(searchstr, singletablecontent, 0, re.M | re.I | re.U)
            if len(splittable) < 3 or isCusHead == False:
                return singletablecontent
            newtable4 = self._modify_long_table_head(
                splittable[2], self.tables["headtype"]
            )
            singletablecontent = (
                splittable[0] + r"\begin{longtable}" + splittable[1] + newtable4
            )
            return singletablecontent

        if self.tables["rowtype"] != "":
            splittable[0] += self.tables["rowtype"] + "\n"

        if isCusHead == True:
            newtable4 = self._modify_table_head(splittable[4], self.tables["headtype"])
        else:
            newtable4 = splittable[4]

        singletablecontent = (
            splittable[0] + splittable[1] + splittable[2] + splittable[3] + newtable4
        )
        if islongtable:
            singletablecontent = self._modify_table_long_head_tail(singletablecontent)

        return singletablecontent

    def _modify_table_long_head_tail(self, singletablecontent):
        searchstr = r"(\\begin{savenotes}\\sphinxattablestart)"
        splittable = re.search(searchstr, singletablecontent, re.M | re.I | re.U)
        tablefirstline = re.sub(
            r"\\sphinxattablestart",
            r"\\sphinxatlongtablestart",
            splittable.group(0),
            re.M | re.I | re.U,
        )
        searchstr = r"(\\begin{tabular}|\\begin{tabulary})(\[[a-z]\]|{\\linewidth}\[[a-z]\])([\s\S].*)"
        splittable = re.search(searchstr, singletablecontent, re.I | re.U)
        headlastpos = splittable.end()
        tablesecondline = re.sub(
            r"\\begin{tabular}|\\begin{tabulary}",
            r"\\begin{longtable}",
            splittable.group(0),
            re.I | re.U,
        )
        tablesecondline = re.sub(r"\{\\linewidth\}", r"", tablesecondline, re.I | re.U)

        searchstr = r"\\sphinxcaption([\s\S].*)"
        splittable = re.search(searchstr, singletablecontent, re.I | re.U)
        longcaption = re.sub(
            r"\\sphinxcaption", r"\\caption", splittable.group(0), re.I | re.U
        )
        longcaption += r"\\*[\sphinxlongtablecapskipadjust]"

        longhead = (
            tablefirstline
            + tablesecondline
            + "\n"
            + r"\sphinxthelongtablecaptionisattop"
            + "\n"
            + longcaption
            + "\n"
        )

        newtablecontent = singletablecontent[headlastpos : len(singletablecontent)]
        endtable = re.sub(
            r"(\\end{tabular}|\\end{tabulary})",
            r"\\end{longtable}",
            newtablecontent,
            re.M | re.I | re.U,
        )
        endtable = re.sub(r"\\par", r"", endtable, re.M | re.I | re.U)
        endtable = re.sub(
            r"\\sphinxattableend",
            r"\\sphinxatlongtableend",
            endtable,
            re.M | re.I | re.U,
        )

        singletablecontent = longhead + endtable
        return singletablecontent

    def _modify_long_table_head(self, content, headtype):
        searchstr = r"\\hline(?P<content>[\s\S]*?)\\hline"
        pattern = re.compile(searchstr, re.M | re.I | re.U)
        matchiter = pattern.finditer(content)
        posarr = []
        i = 0
        for m in matchiter:
            if i > 1:
                break
            posarr.append([])
            posarr[i] = m.span()
            if i == 0:
                newcontent = content[0 : posarr[i][0]]
            else:
                newcontent = newcontent + content[posarr[i - 1][1] : posarr[i][0]]
            newcontent += r"\hline\rowcolor" + headtype
            headcontent = m.group(1)
            if "multicolumn" in headcontent:
                return content
            headlist = []
            if r"\sphinxstyletheadfamily" in headcontent:
                pattern = re.compile(
                    r"(?<=\\sphinxstyletheadfamily)(?P<value>[\s\S]*?)(?=(\\unskip|&)|\\\\)",
                    re.M | re.I | re.U,
                )
                aftercontent = headcontent
                mobjarr = pattern.finditer(aftercontent)

                preposlist = []
                for mobj in mobjarr:
                    amarr = mobj.group("value")
                    curposlist = mobj.span()

                    fontcolor = self.tables["headfontcolor"]
                    amarr = amarr.strip().strip("\r").strip("\n").strip()
                    if amarr == "":
                        continue
                    fontcolor = fontcolor.replace("{}", "{" + amarr + "}", 1)
                    if len(preposlist) > 0:
                        headlist.append(headcontent[preposlist[1] : curposlist[0]])
                    else:
                        headlist.append(headcontent[0 : curposlist[0]])
                    headlist.append(fontcolor)
                    preposlist = curposlist
                headlist.append(headcontent[preposlist[1] : len(headcontent)])
                headcontent = ""
                for prelist in headlist:
                    headcontent = headcontent + prelist + "\n"
                newcontent += headcontent + r"\hline"
            i += 1
        newcontent += content[posarr[i - 1][1] : len(content)]
        return newcontent

    def _modify_table_head(self, content, headtype):
        searchstr = r"\\hline(?P<content>[\s\S]*?)\\hline"
        m = re.search(searchstr, content, re.M | re.I | re.U)
        headcontent = m.group(1)
        posarr = m.span(1)

        if "multicolumn" in headcontent:
            return content

        if r"\sphinxstyletheadfamily" in headcontent:
            pattern = re.compile(
                r"(?<=\\sphinxstyletheadfamily)(?P<value>[\s\S]*?)(?=(\\unskip|&)|\\\\)",
                re.M | re.I | re.U,
            )
            aftercontent = headcontent
        else:
            aftercontent = headcontent.replace(r"\\", "&", 1)
            pattern = re.compile(r"(?P<value>[\s\S]*?)(&\s{1})", re.M | re.I | re.U)

        mobjarr = pattern.finditer(aftercontent)
        headlist = []
        preposlist = []
        for mobj in mobjarr:
            amarr = mobj.group("value")
            curposlist = [mobj.start(), mobj.start() + len(amarr)]

            fontcolor = self.tables["headfontcolor"]
            amarr = amarr.strip().strip("\r").strip("\n").strip()
            if amarr == "":
                continue
            fontcolor = fontcolor.replace("{}", "{" + amarr + "}", 1)
            if len(preposlist) > 0:
                headlist.append(headcontent[preposlist[1] : curposlist[0]])
            else:
                headlist.append(headcontent[0 : curposlist[0]])
            headlist.append(fontcolor)
            preposlist = curposlist

        headlist.append(headcontent[preposlist[1] : len(headcontent)])
        headcontent = ""
        for prelist in headlist:
            headcontent = headcontent + prelist + "\n"
        newcontent = (
            content[0 : posarr[0]]
            + r"\rowcolor"
            + headtype
            + "\n"
            + headcontent
            + content[posarr[1] : len(content)]
        )
        return newcontent

    def _modify_vertical_table(self, singletablecontent):
        searchstr = r"(?<=\\hline)(?P<content>[\s\S]*?)(?=\\hline)"
        pattern = re.compile(searchstr, re.M | re.I | re.U)
        matchiter = pattern.finditer(singletablecontent)
        posarr = []
        i = 0
        for m in matchiter:
            posarr.append([])
            posarr[i] = m.span()
            if i == 0:
                newcontent = singletablecontent[0 : posarr[i][0]]
            else:
                newcontent = (
                    newcontent + singletablecontent[posarr[i - 1][1] : posarr[i][0]]
                )
            cellcontent = m.group("content")
            firstcellcontent = self._modify_first_column_type(cellcontent)
            newcontent += firstcellcontent
            i += 1
        newcontent += singletablecontent[posarr[i - 1][1] : len(singletablecontent)]
        return newcontent

    def _modify_first_column_type(self, cellcontent):
        new_cellcontent = ""
        aftercontent = cellcontent.replace(r"\\", "&", 1)
        aftercontent = aftercontent.strip().strip("\r").strip("\n").strip()

        tmplist = re.split(r"&", aftercontent)

        preposlist = 0
        onelist = tmplist[0]

        fontcolor = self.tables["headfontcolor"]
        onelist = onelist.strip().strip("\r").strip("\n").strip()
        if (r"\textbf" or r"\textcolor") in onelist:
            return cellcontent
        new_cellcontent = (
            "\n"
            + r"\cellcolor"
            + self.tables["headtype"]
            + r"{"
            + fontcolor.replace("{}", "{" + onelist + "}", 1)
            + r"}"
            + "\n"
        )

        for i in range(1, len(tmplist)):
            if len(tmplist[i]) > 0:
                new_cellcontent += "&" + tmplist[i]
        new_cellcontent += r"\\"
        return new_cellcontent + "\n"


def modify_latex(latex_path, latex_content):
    jsonfile = os.path.join(os.path.split(os.path.realpath(__file__))[0], "conf.json")
    with codecs.open(jsonfile, "r+", encoding="utf-8") as load_f:
        json_dict = json.load(load_f)

    for i in range(0, len(latex_content)):
        if os.path.exists("./chapterlogo.pdf"):
            shutil.copy("./chapterlogo.pdf", latex_path)
        filename = latex_content[i][1]
        if filename is None:
            continue
        tex_file = os.path.abspath(os.path.join(latex_path, filename))
        if not os.path.exists(tex_file):
            continue
        fo = codecs.open(tex_file, "r+", encoding="utf-8")
        tex_content = fo.read()
        fo.close

        ModTexobj = ModifyLatex(tex_content, json_dict)
        ModTexobj.replace_package()
        ModTexobj.modify_table_attributes()

        fw = codecs.open(tex_file, "w+", encoding="utf-8")
        fw.write(ModTexobj.content)
        fw.close
