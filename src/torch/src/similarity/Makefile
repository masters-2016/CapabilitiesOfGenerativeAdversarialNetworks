IMAGE_EXTENSION := png

default: data.csv

datasets = $(shell find . ! -path . -type d)
input_images = $(wildcard *.$(IMAGE_EXTENSION))

csv_files = $(foreach dataset,$(datasets), \
	$(foreach image, $(input_images), $(dataset)/$(basename $(image)).csv))

%.csv:
	$(eval DATASET = $(dir $@))
	$(eval INPUT_NAME = $(basename $(notdir $@)))
	$(eval DATASET_NOSLASH = $(shell echo $(basename $(DATASET)) | head -c -2))
	@th compute_similarities.lua $(DATASET) $(INPUT_NAME).$(IMAGE_EXTENSION) \
		$(DATASET_NOSLASH)_$(INPUT_NAME)_out.$(IMAGE_EXTENSION) l2 \
		| grep -E 'min|mean|max' \
		| awk '{print $$2}' \
		| head -c -1 \
		| tr '\n' ',' \
		| (echo -n "$(DATASET_NOSLASH)_$(INPUT_NAME)," && cat) > $@
	@echo "" >> $@

data.csv: $(csv_files)
	cat $(csv_files) > data.csv

data.tex: #data.csv
	@echo '\\begin{tabular}{ | l | c | c | c | c | c | c | }' > $@
	@while read line ; do \
		echo "\t$$line"	| sed -r -e 's/,/ \& /g' \
				| sed -r -e 's|$$| //|' ; \
	done < data.csv >> $@
	@echo "\\end{tabular}" >> $@

clean:
	-@rm *_out.$(IMAGE_EXTENSION)
	-@rm */*.csv
	-@rm data.csv

