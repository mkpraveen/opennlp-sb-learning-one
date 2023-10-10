package com.praveenmk.nlp.controller;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import opennlp.tools.doccat.BagOfWordsFeatureGenerator;
import opennlp.tools.doccat.DoccatFactory;
import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.doccat.DocumentSampleStream;
import opennlp.tools.doccat.FeatureGenerator;
import opennlp.tools.tokenize.TokenSample;
import opennlp.tools.tokenize.TokenSampleStream;
import opennlp.tools.tokenize.TokenizerFactory;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.TrainingParameters;
import opennlp.tools.util.model.ModelUtil;

@RestController
public class OpenNLPController {

	@GetMapping(value = "/train-model")
	public String getTrainModel() throws IOException {

		// Read file with examples of tokenization.
		InputStreamFactory inputStreamFactory = new MarkableFileInputStreamFactory(new File("tokenizerdata.txt"));
		ObjectStream<String> lineStream = new PlainTextByLineStream(inputStreamFactory, StandardCharsets.UTF_8);
		ObjectStream<TokenSample> tokenSampleStream = new TokenSampleStream(lineStream);

		// Train a model from the file read above
		TokenizerFactory tokenizerFactory = new TokenizerFactory("en", null, false, null);
		TokenizerModel tokenizerModel = TokenizerME.train(tokenSampleStream, tokenizerFactory,
				TrainingParameters.defaultParams());
 
		// Serialize model to some file so that next time we don't have to again train a
		// model. Next time We can just load this file directly into model.
		tokenizerModel.serialize(new File("tokenizermodel.bin"));

		// Read file with classifications samples of sentences.
		inputStreamFactory = new MarkableFileInputStreamFactory(
				new File("commodityCategoryData.txt"));
		lineStream = new PlainTextByLineStream(inputStreamFactory, StandardCharsets.UTF_8);
		ObjectStream<DocumentSample> docSampleStream = new DocumentSampleStream(lineStream);

		// Use CUT_OFF as zero since we will use very few samples.
		// BagOfWordsFeatureGenerator will treat each word as a feature. Since we have
		// few samples, each feature/word will have small counts, so it won't meet high
		// cutoff.
		TrainingParameters params = ModelUtil.createDefaultTrainingParameters();
		params.put(TrainingParameters.CUTOFF_PARAM, 0);
		DoccatFactory docCatFactory = new DoccatFactory(new FeatureGenerator[] { new BagOfWordsFeatureGenerator() });

		// Train a model with classifications from above file.
		DoccatModel docCatModel = DocumentCategorizerME.train("en", docSampleStream, params, docCatFactory);

		// Serialize model to some file so that next time we don't have to again train a
		// model. Next time We can just load this file directly into model.
		docCatModel.serialize(new File("documentcategorizer.bin"));

		return new String();
	}

	@GetMapping(value = "/commodity-category")
	public String getCommodityCategory(@RequestParam String shipmentDesc) throws IOException {

	
		String category = "UNKNOWN";
		/**
		 * Load model from serialized file & lets categorize reviews.
		 */
		// Load serialized trained model
		try (InputStream modelIn = new FileInputStream("documentcategorizer.bin")) {

			DoccatModel model = new DoccatModel(modelIn);

			// Initialize document categorizer tool
			DocumentCategorizerME myCategorizer = new DocumentCategorizerME(model);

			// Get the probabilities of all outcome i.e. positive & negative
			double[] probabilitiesOfOutcomes = myCategorizer.categorize(getTokens(shipmentDesc));
			System.out.println("probabilitiesOfOutcomes : " + Arrays.toString(probabilitiesOfOutcomes));
			// Get name of category which had high probability
			category = myCategorizer.getBestCategory(probabilitiesOfOutcomes);
			System.out.println("Category: " + category);

		} catch (Exception e) {
			e.printStackTrace();
		}

		return shipmentDesc + " [ " + category + "]";
	}


	/**
	 * Tokenize sentence into tokens.
	 * 
	 * @param sentence
	 * @return
	 */
	private static String[] getTokens(String sentence) {
 
		// Use model that was created in earlier tokenizer tutorial
		try (InputStream modelIn = new FileInputStream("tokenizermodel.bin")) {
			
			TokenizerME myCategorizer = new TokenizerME(new TokenizerModel(modelIn));
			return myCategorizer.tokenize(sentence); 
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
}
