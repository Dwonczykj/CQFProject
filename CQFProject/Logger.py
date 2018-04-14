import io

def printf(str):
    logfile = open("Cqflog1.txt", 'a')
    print(str)
    logfile.writelines(str)
    logfile.write('\n-------------------------------\n')
    logfile.close()

def convertToLaTeX(df, alignment="c", horAlignment="l"):
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
    output.write("& %s\\\\\\hhline{|%s}\n" % (" & ".join(columnLabels),"=|"*(numColumns + 1)))
    #Write data lines
    for i in range(numRows):
        ind = df.index[i]
        output.write("\\textbf{%s} & %s\\\\\n"
                     % (ind, " & ".join([str(val) for val in df.ix[ind]])))
    #Write footer
    output.write("\\hline\n\end{tabular}\n\end{center}")
    return output.getvalue()

