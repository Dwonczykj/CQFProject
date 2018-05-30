import io
import os
import time
import plotting

Extension = ".tex"

def printf(str):
    logfile = open("Cqflog1.txt", 'a')
    print(str)
    logfile.writelines(str)
    logfile.write('\n-------------------------------\n')
    logfile.close()

def convertToLaTeX(df,name="", alignment="c", horAlignment="l", topLeftCellText=""):
    """
    Convert a pandas dataframe to a LaTeX tabular.
    Prints labels in bold, does not use math mode
    """
    numColumns = df.shape[1]
    numRows = df.shape[0]
    output = io.StringIO()
    colFormat = ("%s|%s" % (alignment, alignment * numColumns)) 
    newColFormat = ("|%s%s" % ((horAlignment + "|") * numColumns, (alignment + "|") * numColumns))
    #Write header
    output.write("\\begin{center}\n \\begin{tabular}{%s}\n\hline\n" % newColFormat)
    columnLabels = ["\\textbf{%s}" % label for label in df.columns]
    output.write("\\textbf{{0}} ".format(topLeftCellText)+"& %s\\\\\\hhline{|%s}\n" % (" & ".join(columnLabels),"=|"*(numColumns + 1)))
    #Write data lines
    for i in range(numRows):
        ind = df.index[i]
        output.write("\\textbf{%s} & %s\\\\\n"
                     % (ind, " & ".join([str(val) for val in df.ix[ind]])))
    #Write footer
    output.write("\\hline\n\end{tabular}\n\end{center}")

    SubmissionFilePath= plotting.SubmissionFilePath #os.path.join('C:\\', 'Users', 'Joe.Dwonczyk', 'Documents', 'CQF', 'CVASection', 'Submission') + "\\CDSBasketProj\\" + time.strftime("%Y%m%d-%H%M%S") + "\\"
    if not os.path.exists(SubmissionFilePath):
        os.makedirs(SubmissionFilePath)
    
    SubmissionFilePath += (name.replace(" ","_").replace(".",",") if not name == "" else "latextable")
    if not os.path.exists(SubmissionFilePath+Extension):
        SubmissionFilePath += Extension
        file = open(SubmissionFilePath,"w")
        file.write(output.getvalue())
                
    else:
        j = 1
        tname = SubmissionFilePath
        while os.path.exists(tname+Extension):
            tname = name + "_%d"%(j)
            j +=1
        SubmissionFilePath = tname + Extension
        file = open(SubmissionFilePath,"w")
        file.write(output.getvalue())
    file.close()

    return output.getvalue()

